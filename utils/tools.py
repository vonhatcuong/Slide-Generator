import json
import os
import logging
import asyncio
import aiohttp
import time # Added time import
from utils.search import Searxng
from models.LLMs import GPT_4o, GPT_o3
from langchain.tools import StructuredTool
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import datetime
from typing import Optional, Annotated, List, Dict # Added List, Dict
from langgraph.prebuilt import InjectedState
from langchain.tools import tool
# from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.getcwd(), "semi_output")
GENERATED_SLIDES_DIR = os.path.join(os.getcwd(), "generated_slides")
LLM = GPT_4o()
gpt_o3 = GPT_o3()

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GENERATED_SLIDES_DIR, exist_ok=True)

# Add argument schemas
class SearchQuery(BaseModel):
    search_query: str
    
class UrlQuery(BaseModel):
    url: str
    
class PresentationOutlineQuery(BaseModel):
    topic: str
    instructions: str

# Helper function for image URL validation
async def _validate_image_url(session: aiohttp.ClientSession, original_item: dict, timeout: int = 5) -> Optional[dict]:
    url = original_item.get("img_src", "")
    if not url:
        return None

    # Normalize URL
    if url.startswith("//"):
        url = "https:" + url
    elif not url.startswith(("http://", "https://")):
        # For robustness, try prepending https. Consider if this is too broad or if such URLs should be skipped.
        url = "https://" + url

    normalized_item = {
        "title": original_item.get("title", ""),
        "content": original_item.get("content", ""), # This content is from Searxng, not the image itself
        "img_src": url, # Use the normalized URL
        "resolution": original_item.get("resolution", "")
    }

    try:
        # Try HEAD request first
        async with session.head(url, timeout=timeout, allow_redirects=True) as response:
            if response.status == 200 and response.content_type.startswith('image/'):
                logger.debug(f"HEAD validated: {url}")
                return normalized_item
    except asyncio.TimeoutError:
        logger.debug(f"HEAD request timed out for {url}")
    except aiohttp.ClientError as e: # More specific exception for client errors
        logger.debug(f"HEAD request failed for {url} (ClientError): {e}")
    except Exception as e: # Catch other potential errors like invalid URL structures before request
        logger.debug(f"HEAD request failed for {url} (General Exception): {e}")

    # If HEAD fails or doesn't confirm, try GET (but don't download full body if possible)
    try:
        async with session.get(url, timeout=timeout, allow_redirects=True) as response:
            if response.status == 200 and response.content_type.startswith('image/'):
                logger.debug(f"GET validated: {url}")
                # Ensure to release the connection if you're not reading the body fully
                await response.release()
                return normalized_item
            else:
                logger.warning(f"GET validation failed for {url} - Status: {response.status}, Content-Type: {response.content_type}")
                await response.release()
                return None
    except asyncio.TimeoutError:
        logger.debug(f"GET request timed out for {url}")
        return None
    except aiohttp.ClientError as e:
        logger.error(f"GET request failed for {url} (ClientError): {e}")
        return None
    except Exception as e:
        logger.error(f"GET request failed for {url} (General Exception): {e}")
        return None

@tool
async def image_search(search_query: str) -> dict: # Changed to async def
    """
    Search for images based on a query, and validate URLs asynchronously.
    
    Args:
        search_query: The query string to search for images
        
    Returns:
        Dictionary with search results including validated image URLs
    """
    logger.info(f"Starting async image search for query: {search_query}")
    tool_start_time = time.perf_counter()
    try:
        searcher: Searxng = Searxng() # This part remains synchronous

        searxng_start_time = time.perf_counter()
        logger.info(f"image_search: Calling Searxng().image_search for query: {search_query}")
        # Fetch more results to account for validation failures, e.g., 20. Kept 10 as per original for now.
        results = json.loads(
            searcher.image_search(search_query, max_results=10)
        )["results"]
        searxng_duration = time.perf_counter() - searxng_start_time
        logger.info(f"image_search: Searxng().image_search completed in {searxng_duration:.2f} seconds.")
        
        validated_results = []
        async with aiohttp.ClientSession() as session:
            validation_tasks = []
            for item in results:
                validation_tasks.append(_validate_image_url(session, item))

            gather_start_time = time.perf_counter()
            logger.info(f"image_search: Starting asyncio.gather for {len(validation_tasks)} URL validations.")
            # Gather results, exceptions are returned as part of the list
            processed_items = await asyncio.gather(*validation_tasks, return_exceptions=True)
            gather_duration = time.perf_counter() - gather_start_time
            logger.info(f"image_search: asyncio.gather for URL validations completed in {gather_duration:.2f} seconds.")

            for item_or_error in processed_items:
                if isinstance(item_or_error, dict) and item_or_error is not None:
                    validated_results.append(item_or_error)
                elif item_or_error is not None: # Log errors if any, but don't include them in results
                    logger.warning(f"Error validating image URL or URL is not an image: {item_or_error}")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_record = {
            "timestamp": timestamp,
            "query": search_query,
            "results": validated_results # Now contains only validated and structured items
        }

        # Saving to file remains synchronous
        output_file_path = os.path.join(OUTPUT_DIR, "image_search_record.json")
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(image_record, f, indent=4) # Added indent for readability
        except IOError as e:
            logger.error(f"Failed to write image search record to {output_file_path}: {e}")

        tool_duration = time.perf_counter() - tool_start_time
        logger.info(f"Async image_search tool execution completed in {tool_duration:.2f} seconds. Found {len(validated_results)} valid images from {len(results)} initial candidates for query '{search_query}'.")
        
        return image_record
    
    except Exception as e:
        tool_duration = time.perf_counter() - tool_start_time
        logger.error(f"Error in async image_search for query '{search_query}' after {tool_duration:.2f} seconds: {str(e)}", exc_info=True)
        return {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "query": search_query, "results": []}

@tool
def web_search(search_query: str, ) -> dict:
    """
    Search for web content based on a query.
    
    Args:
        search_query: The query string to search for web content
        searcher: Searxng instance to use for searching
        
    Returns:
        Dictionary with search results including URLs and content
    """
    logger.info(f"Starting web_search for query: {search_query}")
    tool_start_time = time.perf_counter()
    try:
        searcher: Searxng = Searxng()

        searxng_start_time = time.perf_counter()
        logger.info(f"web_search: Calling Searxng().webpage_search for query: {search_query}")
        full_results = json.loads(
            searcher.webpage_search(search_query, max_results=10)
        )["results"]
        searxng_duration = time.perf_counter() - searxng_start_time
        logger.info(f"web_search: Searxng().webpage_search completed in {searxng_duration:.2f} seconds.")
        
        # Extract only the url, title, content, and score fields
        filtered_results = []
        for result in full_results:
            filtered_result = {
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0)
            }
            filtered_results.append(filtered_result)
        
        
        # Create a search record with timestamp and query
        search_record = {
            "query": search_query,
            "results": filtered_results
        }
        # File I/O for record saving
        output_file_path = os.path.join(OUTPUT_DIR, "web_search_record.json")
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(search_record, f, indent=4)
        except IOError as e:
            logger.error(f"Failed to write web search record to {output_file_path}: {e}")

        tool_duration = time.perf_counter() - tool_start_time
        logger.info(f"web_search tool execution completed in {tool_duration:.2f} seconds. Found {len(filtered_results)} results for query '{search_query}'.")
        return search_record
    
    except Exception as e:
        tool_duration = time.perf_counter() - tool_start_time
        logger.error(f"Error in web_search for query '{search_query}' after {tool_duration:.2f} seconds: {str(e)}", exc_info=True)
        return {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "query": search_query, "results": []}

async def _perform_crawl(session: aiohttp.ClientSession, url: str) -> dict:
    """Helper function to perform a single URL crawl asynchronously."""
    logger.info(f"_perform_crawl: Starting URL crawl for: {url}")
    crawl_start_time = time.perf_counter()
    try:
        # Remove any extra quotes from the URL, but ensure it's a string first
        if isinstance(url, str):
            url = url.strip('"\'')
        else:
            # Handle cases where URL might not be a string (e.g. if bad data is passed)
            logger.error(f"Invalid URL type received: {type(url)} for {url}")
            return {"url": str(url), "content": "Failed to crawl: Invalid URL type", "error": "Invalid URL type"}

        async with session.get(url, timeout=10, allow_redirects=True) as response: # Added allow_redirects
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            html_text = await response.text()
        
        # Parsing logic
        soup_start_time = time.perf_counter()
        soup = BeautifulSoup(html_text, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
        soup_duration = time.perf_counter() - soup_start_time
        logger.debug(f"_perform_crawl: BeautifulSoup parsing for {url} completed in {soup_duration:.4f} seconds.")
        
        crawl_record = {
            "url": url,
            "content": cleaned_text[:6000]  # Store the first 6000 characters
        }
        duration = time.perf_counter() - crawl_start_time
        logger.info(f"_perform_crawl: Successfully crawled and parsed {url} in {duration:.2f} seconds.")
        return crawl_record
    except asyncio.TimeoutError:
        duration = time.perf_counter() - crawl_start_time
        logger.error(f"Timeout during crawl for {url} after {duration:.2f} seconds.")
        return {"url": url, "content": f"Failed to crawl the URL: Timeout", "error": "Timeout"}
    except aiohttp.ClientResponseError as e: # Catch HTTP errors specifically
        duration = time.perf_counter() - crawl_start_time
        logger.error(f"HTTP error during crawl for {url} after {duration:.2f} seconds: {e.status} {e.message}")
        return {"url": url, "content": f"Failed to crawl the URL: HTTP {e.status}", "error": str(e)}
    except aiohttp.ClientError as e: # Catch other client-side errors (e.g., connection refused)
        duration = time.perf_counter() - crawl_start_time
        logger.error(f"Client error during crawl for {url} after {duration:.2f} seconds: {str(e)}")
        return {"url": url, "content": f"Failed to crawl the URL: Client error", "error": str(e)}
    except Exception as e:
        duration = time.perf_counter() - crawl_start_time
        logger.error(f"Generic error in _perform_crawl for {url} after {duration:.2f} seconds: {str(e)}", exc_info=True)
        return {"url": url, "content": f"Failed to crawl the URL: {str(e)}", "error": str(e)}

@tool
async def crawl_url(url: str) -> dict:
    """Crawls a single webpage URL asynchronously to extract its text content."""
    tool_start_time = time.perf_counter()
    logger.info(f"crawl_url: Initiating async crawl for single URL: {url}")
    async with aiohttp.ClientSession() as session:
        result = await _perform_crawl(session, url)
    tool_duration = time.perf_counter() - tool_start_time
    logger.info(f"crawl_url: Tool execution for {url} completed in {tool_duration:.2f} seconds.")
    return result

@tool
def crawl_urls_concurrently(urls: List[str]) -> List[dict]:
    """Crawls a list of webpage URLs concurrently to extract their text content."""
    tool_start_time = time.perf_counter()
    if not urls:
        logger.info("No URLs provided to crawl_urls_concurrently.")
        return []
    logger.info(f"crawl_urls_concurrently: Starting concurrent crawl for {len(urls)} URLs.")
    
    async def _crawl_all():
        async with aiohttp.ClientSession() as session:
            tasks = [_perform_crawl(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=False) # Exceptions are handled in _perform_crawl
        return results

    results = []
    run_async_start_time = time.perf_counter()
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_running():
            logger.warning("crawl_urls_concurrently: Detected running asyncio loop. Attempting to schedule via run_coroutine_threadsafe.")
            future = asyncio.run_coroutine_threadsafe(_crawl_all(), loop)
            results = future.result(timeout=len(urls) * 15)
        else:
            results = asyncio.run(_crawl_all())
    except RuntimeError as e:
        if "cannot be called when another asyncio loop is running" in str(e) or \
           "Nesting asyncio.run() is not allowed" in str(e) or \
           " asyncio.run() cannot be called from a running event loop" in str(e):
            logger.error(f"Asyncio loop conflict in crawl_urls_concurrently: {e}. This tool should ideally be async or called from a synchronous context that can manage a new event loop.")
            return [{"url": url, "content": "Failed to crawl due to asyncio loop conflict.", "error": str(e)} for url in urls]
        logger.error(f"RuntimeError in crawl_urls_concurrently: {e}", exc_info=True)
        return [{"url": url, "content": f"Failed to crawl due to runtime error: {e}", "error": str(e)} for url in urls]
    except Exception as e:
        logger.error(f"Unexpected error in crawl_urls_concurrently's asyncio execution: {e}", exc_info=True)
        return [{"url": url, "content": f"Failed to crawl due to unexpected error: {e}", "error": str(e)} for url in urls]

    run_async_duration = time.perf_counter() - run_async_start_time
    logger.info(f"crawl_urls_concurrently: asyncio part (_crawl_all or run_coroutine_threadsafe) completed in {run_async_duration:.2f} seconds.")

    final_results = [res for res in results if isinstance(res, dict)]
    final_results = [res for res in results if isinstance(res, dict)]

    # Optional: Save a single record for the batch crawl
    batch_crawl_file = os.path.join(OUTPUT_DIR, f"batch_crawl_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.json")
    try:
        with open(batch_crawl_file, "w", encoding="utf-8") as f:
           json.dump({"queried_urls": urls, "results": final_results}, f, indent=4)
    except IOError as e:
        logger.error(f"Failed to write batch crawl record to {batch_crawl_file}: {e}")

    logger.info(f"Concurrent crawl completed. Processed {len(final_results)}/{len(urls)} URLs.")
    return final_results

# # The generate_slide tool that was here has been removed as it's redundant.
# # The primary slide generation logic is within new_workflow.py's slide_agent_node.

# # Create structured tools with args_schema
# # (These are examples and might be outdated or not used if tools are directly decorated with @tool)
# image_search_tool = StructuredTool(
#     name="image_search",
#     description="Search for images based on a query. Returns a list of image URLs.",
#     func=image_search,
#     args_schema=SearchQuery
# )

# web_search_tool = StructuredTool(
#     name="web_search",
#     description="Search for web content based on a query. Returns a list of search results.",
#     func=web_search,
#     args_schema=SearchQuery
# )

# crawl_tool = StructuredTool(
#     name="crawl_url",
#     description="Crawl a webpage URL to extract its text content. Use this when you need to get detailed information from a specific webpage.",
#     func=crawl_url,
#     args_schema=UrlQuery
# )

# generate_presentation_outline_tool = StructuredTool(
#     name="generate_presentation_outline",
#     description="Generate a presentation outline. Requires a topic and instructions. Returns the outline of the presentation.",
#     func=generate_presentation_outline,
#     args_schema=PresentationOutlineQuery
# )
    
# presentation_tool = StructuredTool(
#     name="generate_presentation",
#     description="Generate an HTML presentation slide. Requires five parameters: slide_number (int), title (string), content (string), layout (string), and style (string). Returns the file path of the generated slide.",
#     func=generate_slide,
#     args_schema=PresentationQuery
# )