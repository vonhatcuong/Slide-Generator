from models.LLMs import GPT_4o, GPT_o3, Gemini, Claude_3_7_Sonnet
from langchain.agents import AgentExecutor, create_react_agent
# from langchain.prompts import PromptTemplate
# from utils.tools import Searxng
# from langgraph_supervisor import create_supervisor, create_handoff_tool
# from langgraph.prebuilt import InjectedState
# from langchain_core.runnables import RunnableConfig
# from langgraph.checkpoint.memory import InMemorySaver
# from utils.custom_output_parser import CustomOutputParser

from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import create_react_agent
import uuid, json, os, time # Added time import
import requests, datetime, logging
from langchain_core.tools import tool

from langgraph.types import Command
from langgraph.graph import MessagesState, END, START
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_core.messages import ToolMessage
from utils.tools import image_search, web_search, crawl_url
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
import pprint

OUTPUT_DIR = os.path.join(os.getcwd(), "semi_output")
GENERATED_SLIDES_DIR = os.path.join(os.getcwd(), "generated_slides")
LLM = Gemini()
LLM_4o = GPT_4o()
LLM_o3 = GPT_o3()
LLM_Claude = Claude_3_7_Sonnet()


langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://cloud.langfuse.com"
)

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://cloud.langfuse.com"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@tool
def extract_design_attributes_from_html(html_content: str, attributes_to_extract: list[str]) -> dict:
    """
    Parses HTML content to extract specified design attributes using an LLM.

    Args:
        html_content: The HTML content of the slide.
        attributes_to_extract: A list of attribute names to extract (e.g.,
                               'background_color', 'primary_text_color', 'heading_font_family').

    Returns:
        A dictionary containing the extracted design attributes.
    """
    logger.info(f"extract_design_attributes_from_html: Starting extraction for attributes: {attributes_to_extract}")
    tool_start_time = time.perf_counter()
    prompt = f"""Given the following HTML content:
    ```html
    {html_content}
    ```
    Extract the following design attributes and return them as a VALID JSON object:
    {attributes_to_extract}

    For example, if attributes_to_extract includes 'primary_color', find the dominant color used for primary elements (like main buttons or highlights).
    If it includes 'heading_font_family', find the font family string used for major headings (e.g., H1, H2).
    Be precise. If an attribute cannot be clearly determined from the provided HTML (e.g., it's using Tailwind classes that imply a color but don't state it directly, or it's purely default browser styling), use 'not_found' as the value for that attribute.
    Only include the requested attributes in your JSON response.

    Example attributes to look for (your list might differ):
    - 'background_color': The primary background color of the slide.
    - 'primary_text_color': The main color used for body text.
    - 'heading_text_color': The color used for main headings.
    - 'primary_accent_color': A key accent color used for highlights or important elements.
    - 'heading_font_family': The font-family CSS value for H1 or H2 tags.
    - 'body_font_family': The font-family CSS value for P tags or main content text.

    Your response MUST be a single JSON object.
    """
    try:
        llm_call_start_time = time.perf_counter()
        logger.info("extract_design_attributes_from_html: Calling LLM_Claude.invoke.")
        response = LLM_Claude.invoke(prompt, config=config) # Using LLM_Claude as it's good with JSON
        llm_call_duration = time.perf_counter() - llm_call_start_time
        logger.info(f"extract_design_attributes_from_html: LLM_Claude.invoke completed in {llm_call_duration:.2f} seconds.")

        content = response.content if hasattr(response, 'content') else str(response)
        logger.debug(f"LLM response for attribute extraction: {content}")

        # Attempt to find JSON block if markdown is used
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()

        extracted_attrs = json.loads(content)
        tool_duration = time.perf_counter() - tool_start_time
        logger.info(f"extract_design_attributes_from_html: Successfully extracted attributes in {tool_duration:.2f} seconds: {extracted_attrs}")
        return extracted_attrs
    except json.JSONDecodeError as e:
        tool_duration = time.perf_counter() - tool_start_time
        logger.error(f"JSONDecodeError in extract_design_attributes_from_html after {tool_duration:.2f} seconds: {e}. LLM Output: {content}")
        return {attr: "not_found_due_to_json_error" for attr in attributes_to_extract}
    except Exception as e:
        tool_duration = time.perf_counter() - tool_start_time
        logger.error(f"Error in extract_design_attributes_from_html after {tool_duration:.2f} seconds: {e}. LLM Output: {content}", exc_info=True)
        return {attr: "not_found_due_to_error" for attr in attributes_to_extract}

class AgentState(TypedDict):
    messages: List[BaseMessage]
    outline: str
    images: list[dict]
    found_information: list[dict]
    input: str
    slides: list[dict]
    summary: list[dict]
    extracted_design_attributes: dict # Added for storing design attributes

members = ["outline_agent", "slide_agent", "summarizer"]
options = members + ["FINISH"]

class Router(TypedDict):
    next: Literal["outline_agent", "slide_agent", "summarizer", "FINISH"]

class Slide(TypedDict):
    slide_number: int
    summary: str
    need_enhance_visual: bool

class Summarize(TypedDict):
    slides: list[Slide]

def supervisor_node(state: AgentState) -> Command[Literal["outline_agent", "slide_agent", "summarizer", "__end__"]]:
    """
    Router function that decides which agent should run next based on the current state.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command indicating which node to go to next
    """
    logger.info("Supervisor node: Starting workflow routing")

    last_message = state["messages"][-1] if state["messages"] else None

    # Deterministic routing after outline_agent
    if hasattr(last_message, 'name') and last_message.name == "outline_agent":
        if state.get("outline") and state["outline"]: # Check if outline exists and is not empty
            logger.info("Supervisor node: Deterministic route from outline_agent to slide_agent")
            return Command(goto="slide_agent", update={"next": "slide_agent"})
        else:
            logger.warning("Supervisor node: Outline agent did not produce an outline. Falling back to LLM.")
            # Fall through to LLM-based routing for error handling or unexpected state

    # Deterministic routing after slide_agent
    if hasattr(last_message, 'name') and last_message.name == "slide_agent":
        if state.get("slides") and state["slides"]: # Check if slides list exists and is not empty
            logger.info("Supervisor node: Deterministic route from slide_agent to summarizer")
            return Command(goto="summarizer", update={"next": "summarizer"})
        else:
            logger.warning("Supervisor node: Slide agent did not produce slides. Falling back to LLM.")
            # Fall through to LLM-based routing

    # LLM-based routing for initial calls, after summarizer, or fallbacks
    logger.info("Supervisor node: Using LLM for routing decision.")
    if state.get("summary"): # Use .get for safer access
        summary_info = f"This is the summary of the presentation: {state['summary']}"
    else:
        summary_info = "No summary has been generated yet."
    
    # Keep system prompt largely the same, as LLM still needs full context for its decisions
    system_prompt = f"""
    You are a supervisor, tasked with managing a conversation between the following workers: {members}.
    The last message was: {last_message.content if last_message else 'None'}. Current state outline: {'Exists' if state.get('outline') and state.get('outline') else 'Missing'}. Current state slides: {'Exists' if state.get('slides') and state.get('slides') else 'Missing'}.
    You can respond to the user's general questions, then go to end.
    Given the user's request and the current state, respond with the worker to act next.
    {summary_info}

    Workflow reminder:
    - Normally, if an outline is ready, the next step is 'slide_agent'. (This might be handled deterministically now)
    - Normally, if slides are generated, the next step is 'summarizer'. (This might be handled deterministically now)
    - If a slide summary is available, you need to decide if slides need enhancement (not implemented yet, so usually FINISH) or if the process should FINISH.
    - If any agent fails or provides unexpected output, decide the best course of action (e.g., retry, FINISH, or ask user).

    When finished creating the presentation, or if instructed to stop, or if critical errors occur, go to FINISH.

    Respond with ONLY one of these options: {', '.join(options)}
    """

    llm_messages = [{"role": "system", "content": system_prompt}]
    llm_messages.extend(state["messages"][-5:]) # Send last 5 messages for context

    response = LLM_4o.invoke(llm_messages, config=config)
    goto = response.content.strip()
    
    if goto not in options:
        logger.warning(f"Invalid response from supervisor: {goto}. Defaulting based on state.")
        if not state.get("outline") or not state.get("outline"): # Check if outline is missing or empty
            goto = "outline_agent"
        elif not state.get("slides") or not state.get("slides"): # Check if slides are missing or empty
            goto = "slide_agent"
        elif not state.get("summary"):
            goto = "summarizer"
        else:
            goto = "FINISH"
        
    if goto == "FINISH":
        goto = END # Ensure using the correct graph end state

    logger.info(f"Supervisor node: LLM routed to {goto}")
    return Command(goto=goto, update={"next": goto})

def outline_agent_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Agent that generates the presentation outline.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command to update the state and proceed to supervisor
    """
    logger.info("outline_agent_node: Starting outline generation process.")
    agent_invoke_start_time = time.perf_counter()
    
    prompt = """You are a research assistant helping to create a presentation. Follow these steps:

    1. First, use `web_search` to gather information about the topic. This will give you a list of relevant URLs and snippets.
    2. Review the results from `web_search`. Identify specific websites that seem most promising for detailed information.
    3. To get detailed information from a single specific website, use `crawl_url` (Note: this tool is now asynchronous, the system will handle the await).
    4. If you have identified MULTIPLE important URLs from `web_search` that you need to fetch content from, you can use `crawl_urls_concurrently`. Provide this tool with a list of these URLs (e.g., `["url1", "url2", "url3"]`) to fetch their content more efficiently.
    5. After gathering information, use `image_search` to find relevant images for the presentation (Note: this tool is also asynchronous).
    6. Finally, generate the full presentation content based on all the gathered information (text from web searches, crawled content, and image details) and follow the user's instructions for the presentation structure.

    Use all the gathered information to generate the presentation content.
    The presentation content should contains these slide:
    - Cover slide
    - Table of contents
    - Introduction slide
    - Main content slides
    - Key points slides
    - Graphs and charts slides
    - Conclusion slide
    - Reference slide
    
    If the user ask for 5 slides of presentation for example, you should exclude cover slide and table of contents slide, make sure table of contents slide cover all the slides.
    
    IMPORTANT: You MUST use at least web_search, crawl_url and image_search before generating the outline.
    IMPORTANT: Make the full presentation content with as many words as possible, not just the outline.
    """
    
    # Import the new crawl_urls_concurrently tool
    from utils.tools import crawl_urls_concurrently

    outline_agent = create_react_agent(
        model=LLM,
        tools=[image_search, crawl_url, web_search, crawl_urls_concurrently], # Added crawl_urls_concurrently
        prompt=prompt
    )
    
    logger.info("outline_agent_node: Invoking ReAct agent for outline generation.")
    result = outline_agent.invoke(state, config=config)
    agent_invoke_duration = time.perf_counter() - agent_invoke_start_time
    logger.info(f"outline_agent_node: ReAct agent invocation completed in {agent_invoke_duration:.2f} seconds.")
    
    # Extract tool outputs from messages
    outline = result
    images = []
    found_info = []
    
    # Handle potential missing 'messages' key in result
    try:
        messages_list = result.get("messages", [])
        if not messages_list and hasattr(result, "messages"):
            messages_list = result.messages
    except (AttributeError, TypeError):
        logger.warning(f"Unexpected result format: {type(result)}")
        messages_list = []
        if isinstance(result, list):
            messages_list = result
    
    last_message_content = "Outline generation completed"
    
    for message in messages_list:
        if isinstance(message, ToolMessage):
            logger.debug(f"Processing tool message: {message.name}")
            try:
                content = json.loads(message.content)
                
                if message.name == "image_search":
                    if isinstance(content, dict) and "results" in content:
                        images.extend(content["results"])
                        logger.info(f"Found {len(content['results'])} images")
                    else:
                        images.append({"url": message.content})
                        logger.info("Found 1 image")
                elif message.name in ["web_search", "crawl_url", "crawl_urls_concurrently"]: # Added crawl_urls_concurrently
                    if isinstance(content, dict):
                        # Handle single crawl_url result or web_search result structure
                        if message.name != "crawl_urls_concurrently" and "results" not in content:
                             found_info.append({
                                "source": message.name,
                                "content": content.get("content", ""),
                                "title": content.get("title", "") if message.name == "web_search" else content.get("url")
                            })
                             logger.info(f"Found information from {message.name}: {content.get('title', content.get('url'))}")

                        # Handle web_search results (list of results)
                        elif "results" in content and isinstance(content["results"], list):
                            for res_item in content["results"]:
                                found_info.append({
                                    "source": message.name,
                                    "content": res_item.get("content", ""),
                                    "title": res_item.get("title", "")
                                })
                            logger.info(f"Found {len(content['results'])} results from {message.name}")
                        # Handle crawl_urls_concurrently results (list of results)
                        elif message.name == "crawl_urls_concurrently" and isinstance(content, list): # content is directly the list
                            for res_item in content:
                                found_info.append({
                                    "source": message.name,
                                    "content": res_item.get("content", ""),
                                    "url": res_item.get("url","")
                                })
                            logger.info(f"Found {len(content)} results from {message.name}")
                        else: # Fallback for other dict structures if necessary
                             found_info.append({
                                "source": message.name,
                                "content": str(content) # stringify if unknown structure
                            })
                             logger.info(f"Found information from {message.name} (unknown structure)")
                    elif isinstance(content, list): # Handles direct list output from crawl_urls_concurrently if not wrapped in "results"
                        for res_item in content:
                             found_info.append({
                                "source": message.name,
                                "content": res_item.get("content", ""),
                                "url": res_item.get("url","")
                            })
                        logger.info(f"Found {len(content)} results from {message.name}")
                    else: # Non-dict, non-list content
                        found_info.append({
                            "source": message.name,
                            "content": str(message.content), # stringify the original content
                        })
                        logger.info(f"Found information from {message.name} (non-dict content)")
                        
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from {message.name}: {message.content}")
                # if the content is a list of dicts (from crawl_urls_concurrently) but not valid JSON string
                if message.name == "crawl_urls_concurrently" and isinstance(message.content, str) and message.content.strip().startswith("["):
                    try:
                        # Try to manually parse it if it looks like a list of JSON objects
                        parsed_list = json.loads(message.content)
                        if isinstance(parsed_list, list):
                            for res_item in parsed_list:
                                found_info.append({
                                    "source": message.name,
                                    "content": res_item.get("content", ""),
                                    "url": res_item.get("url","")
                                })
                            logger.info(f"Successfully parsed list from {message.name} after initial JSON error.")
                    except Exception as e_parse:
                        logger.error(f"Could not manually parse list-like content from {message.name}: {e_parse}")
                elif message.name == "image_search":
                    images.append({"url": message.content})
                elif message.name in ["web_search", "crawl_url"]:
                    found_info.append({
                        "source": message.name,
                        "content": str(message.content), # Ensure it's a string
                    })
        
        # Save the last message content for the return
        if hasattr(message, "content"):
            last_message_content = message.content
            
    logger.info(f"last_message_content: {last_message_content}")
            
    return Command(
        update={
            "messages": [HumanMessage(content=last_message_content, name="outline_agent")],
            "outline": outline,
            "images": images,
            "found_information": found_info
        },
        goto="supervisor"
    )
    
def slide_agent_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Agent that generates the presentation slides.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command to update the state and proceed to supervisor
    """
    logger.info("Slide agent: Starting slide generation")
    
    images = state["images"]
    found_info = state["found_information"]
    
    
    @tool
    def generate_slide(slide_number: int, instructions: str, images_url: str, style: str, color_scheme: str, design_language: str, first_slide_reference: str = "", design_attrs: dict = None) -> tuple[str, str]:
        """
        Generate a single HTML slide, save it, and return its content along with a success message.
        
        Args:
            slide_number: The number of the slide to generate
            instructions: The instructions for the slide content
            images_url: The images URL to use for the slide
            style: The style of the slide
            color_scheme: The color scheme of the slide
            design_language: The design language of the slide
            first_slide_reference: The first slide content for design consistency (optional)
            design_attrs: Specific design attributes extracted from the first slide (optional)
        Returns:
            A tuple containing:
                - String with information about the generated slide.
                - The HTML content of the generated slide.
        """
        tool_start_time = time.perf_counter()
        logger.info(f"generate_slide: Starting generation for slide #{slide_number}.")
        try:
            with open("rules/instruction.txt", "r") as f:
                instruction_rules = f.read()

            # Add consistency reference if available
            consistency_instruction = ""
            if first_slide_reference and slide_number > 1:
                consistency_instruction = f"""
                IMPORTANT FOR CONSISTENCY (BROAD): This is the first slide that was generated. Use it as a reference for maintaining consistent design style, color scheme, layout patterns, and overall visual identity for aspects NOT covered by specific design_attrs:
                {first_slide_reference}
                
                Please maintain the same general:
                - Layout structure and spacing
                - Design elements and visual style
                - Overall aesthetic approach
                """

            design_attrs_instruction = ""
            if design_attrs:
                design_attrs_instruction = f"""
                IMPORTANT FOR CONSISTENCY (SPECIFIC): Prioritize the following exact design attributes:
                - Background Color: {design_attrs.get('background_color', 'Refer to instruction_rules or first_slide_reference')}
                - Primary Text Color: {design_attrs.get('primary_text_color', 'Refer to instruction_rules or first_slide_reference')}
                - Heading Text Color: {design_attrs.get('heading_text_color', 'Refer to instruction_rules or first_slide_reference')}
                - Primary Accent Color: {design_attrs.get('primary_accent_color', 'Refer to instruction_rules or first_slide_reference')}
                - Heading Font Family: {design_attrs.get('heading_font_family', 'Refer to instruction_rules or first_slide_reference')}
                - Body Font Family: {design_attrs.get('body_font_family', 'Refer to instruction_rules or first_slide_reference')}
                (Use these provided values directly for these specific CSS properties.)
                These attributes were extracted from the first slide. Use them to ensure strict consistency for these specific elements.
                For other styling aspects not listed here, continue to refer to `instruction_rules` and the broader context of `first_slide_reference`.
                """

            presentation_prompt = f"""
            You are a professional presentation designer.
            Your task is to create a single HTML slide.

            Adhere to the design and style guidelines provided in these instructions:
            {instruction_rules}

            {design_attrs_instruction}

            Content Instructions for this slide: {instructions}
            Available Images: {images_url}
            Overall Presentation Style: {style}
            Color Scheme: {color_scheme} # This should align with instruction_rules' "Required Style Elements" and design_attrs if provided.
            Design Language: {design_language} # This should align with instruction_rules' "Required Style Elements" and design_attrs if provided.
            
            {consistency_instruction}

            Key Requirements:
            - Use Tailwind CSS for styling.
            - Employ Google Fonts and Font Awesome icons as specified in the instruction_rules (unless overridden by specific design_attrs like font family).
            - Ensure the slide is responsive and fits 1280x720px. (width: 1280px; min-height: 720px; position: relative;)
            - Generate detailed and professional content.
            - The HTML should be self-contained.
            - All elements should be wrapped in a div or section that splits the slide into multiple sections if appropriate for layout.
            - Background should be a solid color or subtle gradient as per the design language in instruction_rules (and design_attrs if 'background_color' is present), not a hero image unless it's a title slide and explicitly requested in `instructions`.
            - If the content is short, make it 1 column and in appropriate scale; if the content is long, consider multiple columns for readability.
            - Use chart.js if graphs or charts are needed (avoid `maintainAspectRatio: false` to prevent charts from spreading out).
            - Ensure images are wrapped in a section or div, appropriately scaled (not too big/small, wide/narrow), and responsive.
            - Use large font sizes and bold text for emphasis where needed.
            - Generate as many tokens as possible to create a complete and detailed slide.
            """
            
            # Get the response and extract the content
            llm_call_start_time = time.perf_counter()
            logger.info(f"generate_slide #{slide_number}: Calling LLM_Claude.invoke.")
            response = LLM_Claude.invoke(presentation_prompt, config=config)
            llm_call_duration = time.perf_counter() - llm_call_start_time
            logger.info(f"generate_slide #{slide_number}: LLM_Claude.invoke completed in {llm_call_duration:.2f} seconds.")
            html_content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract only the HTML content between <!DOCTYPE html> and </html>
            start_marker = "<!DOCTYPE html>"
            end_marker = "</html>"
            start_idx = html_content.find(start_marker)
            end_idx = html_content.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                html_content = html_content[start_idx:end_idx + len(end_marker)]
            else:
                # If proper HTML markers are not found, ensure we still have some content to avoid errors
                logger.warning("Could not find proper <!DOCTYPE html> ... </html> markers in the LLM response. Using the raw response.")
                if not html_content.strip(): # if html_content is empty or whitespace
                     html_content = "<div>Error: Empty slide content generated by LLM.</div>"

            
            # Save individual slide
            os.makedirs(GENERATED_SLIDES_DIR, exist_ok=True)
            output_path = os.path.join(GENERATED_SLIDES_DIR, f"slide_{slide_number:03d}.html")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            tool_duration = time.perf_counter() - tool_start_time
            logger.info(f"generate_slide #{slide_number}: Tool execution completed in {tool_duration:.2f} seconds. Saved to {output_path}")
            return f"Slide #{slide_number} generated successfully. Saved to {output_path}", html_content
        except Exception as e:
            tool_duration = time.perf_counter() - tool_start_time
            logger.error(f"Failed to generate slide #{slide_number} after {tool_duration:.2f} seconds: {str(e)}", exc_info=True)
            return f"Failed to generate slide: {str(e)}", f"<div>Error generating slide: {str(e)}</div>"
    
    with open("rules/instruction.txt", "r") as f:
        instruction = f.read()
        
    prompt = f"""You are a presentation slide generator. Your task is to create slides based on the outline.
    
    This is list of images that you can use for the slides: {images}
    This is the general instructions for all slides (including Required Style Elements from instruction.txt):
    {instruction}
    
    Extracted design attributes from the first slide will be available under `state.extracted_design_attributes` after the first slide is generated and processed.

    Strategy:
    1. For each section of the presentation outline, you will generate one slide.
    2. **For the first slide (slide_number=1):**
        a. Call `generate_slide` with appropriate parameters. `first_slide_reference` will be empty. `design_attrs` will be empty.
        b. The `generate_slide` tool will return a message and the HTML content of the first slide.
        c. After the first slide is generated, call `extract_design_attributes_from_html` tool. Provide it the HTML content of the first slide and a list of attributes to extract, for example: `['background_color', 'primary_text_color', 'heading_text_color', 'primary_accent_color', 'heading_font_family', 'body_font_family']`.
        d. The extracted attributes will be stored in the workflow state and become available for subsequent slides. You don't need to directly receive them, but know that they will be used.
    3. **For subsequent slides (slide_number > 1):**
        a. Call `generate_slide` again.
        b. This time, ensure you pass the HTML content of the *first slide* as the `first_slide_reference` parameter.
        c. Also, the system will automatically pass the `extracted_design_attributes` (that you helped extract via the tool call after slide 1) as the `design_attrs` parameter to `generate_slide`. Your generated call to `generate_slide` should account for this parameter being present by its name `design_attrs` if you need to explicitly mention it, but often the system will handle its inclusion if you focus on the other parameters. The key is that the `generate_slide` tool itself will use these `design_attrs` for stricter consistency.
    4. Repeat until all slides based on the outline are generated.

    General Slide Generation Guidelines:
    - Use content from the `outline`, `found_info`, and `images` for each slide.
    - Adhere to design principles in `instruction` and `first_slide_reference`.
    - For slides 2+, `generate_slide` will internally prioritize `design_attrs` for specific styling elements.
    - Ensure each slide is detailed, professional, and responsive.
    - Generate slides one at a time.

    IMPORTANT:
    - Your primary responsibility is to call `generate_slide` for each part of the outline.
    - After `generate_slide` for slide 1, you MUST call `extract_design_attributes_from_html` using the HTML output of slide 1.
    - For `generate_slide` calls for slides 2+, ensure `first_slide_reference` is correctly passed (it's the HTML of slide 1). The `design_attrs` will be handled by the system if you've called the extraction tool correctly.
    """
    
    # Define the tools available to this agent
    available_tools = [generate_slide, extract_design_attributes_from_html]

    slide_agent = create_react_agent(
        model=LLM_Claude,
        tools=available_tools,
        prompt=prompt
    )    
    try:
        # Initialize extracted_design_attributes in state if not present
        if "extracted_design_attributes" not in state:
            state["extracted_design_attributes"] = {}

        logger.info("slide_agent_node: Invoking ReAct agent for slide generation/extraction.")
        agent_invoke_start_time = time.perf_counter()
        result = slide_agent.invoke(state, config=config)
        agent_invoke_duration = time.perf_counter() - agent_invoke_start_time
        logger.info(f"slide_agent_node: ReAct agent invocation completed in {agent_invoke_duration:.2f} seconds.")
        
        # The 'slides' list in AgentState is intended to store a summary or reference for the supervisor/summarizer.
        # The actual first slide HTML for reference and extracted attributes are handled within this node's logic
        # and passed directly or via state to subsequent tool calls within this node's invocation.

        # The result from create_react_agent is a dictionary containing 'messages'.
        # We need to find the relevant ToolMessage to see what happened.
        
        # It's tricky to update 'extracted_design_attributes' here directly if the ReAct agent doesn't explicitly output it.
        # The ReAct agent is responsible for calling the tools. The state updates should reflect the *outcome* of those calls.

        # Let's assume the ReAct agent's execution of 'extract_design_attributes_from_html'
        # will result in a HumanMessage or similar that we might want to log or process.
        # However, the actual update to state.extracted_design_attributes should ideally be handled
        # by the tool itself or a subsequent step if the tool can't directly write to AgentState.
        # For now, we rely on the ReAct framework to manage tool calls and their internal effects if any.
        # The critical part is that the `slide_agent` prompt tells the LLM to call the extraction tool.

        # The `slides` list in the state is used by the supervisor to decide if it should go to summarizer.
        # We need to ensure it's populated, at least with a marker for the first slide.
        # The actual HTML of the first slide for `first_slide_reference` and `extracted_design_attributes`
        # are managed more directly by the agent's internal logic when it calls `generate_slide`.

        current_slides_summary = state.get("slides", [])
        # Check the last message from the agent to see if a slide was generated or attributes extracted.
        # This part is complex because the ReAct agent makes multiple calls.
        # We'll simplify by saying if any slide was generated, the 'slides' list should reflect that for the supervisor.
        # A more robust way would be for the ReAct agent to explicitly output what it did.

        # The ReAct agent is expected to make calls to generate_slide and extract_design_attributes.
        # The key is that the `extracted_design_attributes` in the state is updated by the tool call.
        # The `slides` list in the state is mostly for the supervisor.
        # If generate_slide was called, we assume a slide was made.
        
        # The actual update to 'extracted_design_attributes' will happen if the LLM calls the tool.
        # We need to ensure the ReAct agent is prompted to do so.
        # The current approach updates the state["slides"] which is used by the supervisor.
        # The `extracted_design_attributes` is part of the state passed to the react agent, so it can use it.

        # The main output to update the supervisor about progress is via messages and the 'slides' field.
        # If the agent called 'generate_slide', we can assume 'slides' list should be non-empty for supervisor.
        # The prompt for the ReAct agent needs to be very clear about calling extract_design_attributes_from_html
        # and then using the result (implicitly via state) for subsequent generate_slide calls.

        logger.info(f"Slide agent interaction result: {result}")
        logger.info("Slide agent: Completed current round of slide generation/extraction.")

        agent_messages = result.get("messages", [])
        last_response_message = agent_messages[-1] if agent_messages else HumanMessage(content="Slide agent processing complete.", name="slide_agent")
        
        newly_extracted_attributes = state.get("extracted_design_attributes", {})
        first_slide_html_content_for_reference = state.get("first_slide_html_content", "") # Get existing or empty

        slide_summary_for_supervisor = state.get("slides", []) # Get existing slide summaries

        for msg in reversed(agent_messages): # Check from recent messages
            if isinstance(msg, ToolMessage):
                if msg.name == "extract_design_attributes_from_html":
                    try:
                        # Assuming the tool returns a string representation of a dict, or the dict itself.
                        # If it's a string, parse it. If it's already a dict, use it.
                        if isinstance(msg.content, str):
                            attrs = json.loads(msg.content)
                        elif isinstance(msg.content, dict):
                            attrs = msg.content
                        else:
                            attrs = {}

                        if attrs: # If not empty
                            newly_extracted_attributes = attrs
                            logger.info(f"Captured extracted design attributes from tool call: {newly_extracted_attributes}")
                            # No need to break, but we've found the latest extraction.
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse extracted attributes from ToolMessage content: {msg.content}")
                    except Exception as e:
                        logger.error(f"Error processing ToolMessage for extracted_attributes: {e}")

                elif msg.name == "generate_slide":
                    # The generate_slide tool now returns a tuple (message_str, html_content_str)
                    # The actual ToolMessage.content might be the string representation of this tuple,
                    # or if the ReAct framework handles it well, it might be directly accessible.
                    # For simplicity, we assume the ReAct agent was prompted to get the HTML of slide 1
                    # and then pass it to extract_design_attributes_from_html.
                    # The `first_slide_html_content_for_reference` should be set if slide 1 was made.
                    # This part is tricky as the ReAct agent handles tool outputs internally.
                    # The prompt ensures the agent *knows* to use first slide HTML for reference.
                    # Let's assume the agent's internal state or subsequent calls correctly use first_slide_reference.
                    # We also need to update the 'slides' list for the supervisor.
                    if not slide_summary_for_supervisor: # if slides list is empty, means first slide was likely just processed
                        try:
                            # msg.content from generate_slide is now a tuple string like "('Success...', '<html>...</html>')"
                            # We only need to confirm a slide was made for the supervisor's list.
                            # The actual HTML for reference is handled by the agent's logic (prompting).
                            # Extract slide number if possible from the message string part of the tuple
                            content_tuple_str = msg.content
                            # A bit of a hack to parse the tuple string; proper parsing might be needed
                            if isinstance(content_tuple_str, str) and content_tuple_str.startswith("('") :
                                slide_msg_part = content_tuple_str.split("', '")[0][2:] # Get the message part
                                if "Slide #1" in slide_msg_part or "slide_1" in slide_msg_part : # Check if it's slide 1
                                     slide_summary_for_supervisor = [{"slide_number": 1, "summary": "First slide processed."}]
                                     logger.info("Slide agent node: Marked first slide as processed for supervisor.")
                                     # The actual HTML of slide 1 needs to be available to the agent for `first_slide_reference`
                                     # and for the `extract_design_attributes_from_html` call.
                                     # This relies on the ReAct agent being prompted to handle this flow.
                        except Exception as e:
                            logger.error(f"Could not parse slide number from generate_slide ToolMessage: {e}")


        # If slide_summary_for_supervisor is still empty, but agent was invoked, means something happened.
        if not slide_summary_for_supervisor and agent_messages:
             slide_summary_for_supervisor = [{"slide_number": "unknown", "summary": "Agent processed, but no slide generation detected for summary."}]


        return Command(
            update={
                "messages": [last_response_message],
                "slides": slide_summary_for_supervisor,
                "extracted_design_attributes": newly_extracted_attributes
            },
            goto="supervisor"
        )
    except Exception as e:
        logger.error(f"Error in slide generation: {str(e)}")
        return Command(
            update={
                "messages": [HumanMessage(content=f"Error generating slides: {str(e)}", name="slide_agent")],
                "slides": []
            },
            goto="supervisor"
        )
        
def summarizer_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Agent that summarizes the presentation slides.
    
    Args:
        state: The current state of the workflow
    """
    logger.info("summarizer_node: Starting slide summarization.")
    
    slides = state["slides"]
    prompt = f"""
    You are a summarizer, tasked with summarizing the presentation slides.
    
    This is the list of slides: {slides}
    
    For each slide, analyze if it needs visual enhancement and provide a brief summary.
    Format your response as follows for each slide:
    Slide [number]: [brief summary] - [Needs visual enhancement/Visuals are good]
    
    Example format:
    Slide 1: Introduction to the topic - Visuals are good
    Slide 2: Key concepts overview - Needs visual enhancement
    """
    llm_call_start_time = time.perf_counter()
    logger.info("summarizer_node: Calling LLM.invoke for summarization.")
    response = LLM.invoke(prompt, config=config)
    llm_call_duration = time.perf_counter() - llm_call_start_time
    logger.info(f"summarizer_node: LLM.invoke for summarization completed in {llm_call_duration:.2f} seconds.")
    
    summary_text = response.content.strip()
    logger.info("summarizer_node: Completed all slide summaries.")
    
    return Command(
        update={
            "messages": [HumanMessage(content=summary_text, name="summarizer")],
            "summary": summary_text
        },
        goto="supervisor"
    )


graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("outline_agent", outline_agent_node)
graph.add_node("slide_agent", slide_agent_node)
graph.add_node("summarizer", summarizer_node)
graph.add_edge(START, "supervisor")

app = graph.compile()

trace_id = str(uuid.uuid4())
# Base configuration
config = {
    "recursion_limit": 100,
    "configurable": {
        "trace_id": trace_id
    },
    "callbacks": [langfuse_handler],
    "run_id": trace_id
}

while True:
    user_input = input("\nEnter your query (or 'exit' to quit): ")
    logger.info(f"Received user input: {user_input}")

    if user_input.lower() == 'exit':
        logger.info("User requested exit")
        print("Goodbye!")
        break

    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "outline": "",
        "images": [],
        "found_information": [],
        "input": user_input,
        "slides": [],
        "summary": []
    }
    logger.info("Created initial state")

    try:
        logger.info("Starting workflow execution")
        result = app.invoke(initial_state, config=config)
        logger.info("Workflow execution completed")
        logger.debug(f"Final result: {result}")

        print("\n[DEBUG] Full result:")
        pprint.pprint(result)

        for m in result["messages"]:
            if isinstance(m, ToolMessage):
                logger.debug(f"Tool message: {m.content}")
                print(f"ToolMessage: {m.content}")
    except Exception as e:
        logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
        print(f"Error occurred: {str(e)}")