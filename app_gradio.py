"""
MedMARS Gradio Interface
========================
Simple web interface for MedMARS - Medical Multi-modal Agent with Reasoning and Search

Usage:
    python app_gradio.py
"""

import os
import sys
import json
import gradio as gr
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.medmars import MedMARS
from src.image_patch import ImagePatch


# Initialize MedMARS
print("Initializing MedMARS...")
medmars = MedMARS()
print("MedMARS is ready!")


# Global state to store intermediate results
class SessionState:
    def __init__(self):
        self.image_path = None
        self.output_dir = None
        self.thought = None
        self.plan = None
        self.code = None
        self.execution_output = None
        self.result = None

state = SessionState()


def convert_to_serializable(obj):
    """
    Convert an object into a JSON-serializable format

    Args:
        obj: Object to convert (may be a numpy array, dict, list, etc.)

    Returns:
        JSON-serializable object
    """
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def format_output_as_json(output):
    """
    Pretty-print the output as JSON

    Args:
        output: Output from the code execution

    Returns:
        Formatted string (pretty JSON)
    """
    try:
        # Convert to a JSON-serializable object
        serializable = convert_to_serializable(output)

        # Pretty-print with indentation
        formatted = json.dumps(serializable, indent=2, ensure_ascii=False)

        return formatted

    except Exception:
        # Fall back to pprint when the output cannot be JSON-encoded
        try:
            import pprint
            return pprint.pformat(output, indent=2, width=80)
        except Exception:
            # Last resort: return the raw string
            return str(output)


def step1_planning(image, question):
    """
    STEP 1: Planning - equivalent to the planner stage in medmars.run()

    Mirrors this code in medmars.run():
        self.thought, self.plan = self.planner(query=query, image_path=image_path)
    """
    print("\n" + "="*60)
    print("STEP 1: PLANNING")
    print("="*60)

    if image is None:
        return "❌ Error: no image provided", "", "", ""

    if not question or question.strip() == "":
        return "❌ Error: no question provided", "", "", ""

    try:
        # Create temporary output directory
        state.output_dir = Path("static/temp_gradio")
        state.output_dir.mkdir(parents=True, exist_ok=True)

        # Save image temporarily
        temp_image_path = state.output_dir / "input_image.jpg"
        image.save(str(temp_image_path))
        state.image_path = str(temp_image_path)

        print(f"📝 Question: {question}")
        print(f"🖼️  Image saved to: {state.image_path}")

        # Call planner - same as medmars.run()
        state.thought, state.plan = medmars.planner(query=question, image_path=state.image_path)

        print(f"💭 Thought: {state.thought[:100]}...")
        print(f"📋 Plan: {state.plan[:100]}...")

        return "✅ Step 1: Planning complete", state.thought, state.plan, ""

    except Exception as e:
        print(f"❌ Error in STEP 1: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", "", "", ""


def step2_code_generation():
    """
    STEP 2: Code Generation - equivalent to the code_generator stage in medmars.run()

    Mirrors this code in medmars.run():
        self.code = self.code_generator(self.plan)
    """
    print("\n" + "="*60)
    print("STEP 2: CODE GENERATION")
    print("="*60)

    if not state.plan:
        return "❌ Error: no plan available yet", "", ""

    try:
        # Call code_generator - same as medmars.run()
        state.code = medmars.code_generator(state.plan)

        print(f"💻 Code: {len(state.code)} chars")
        print("Code preview:")
        print(state.code[:200] + "...")

        return "✅ Step 2: Code generated", state.code, ""

    except Exception as e:
        print(f"❌ Error in STEP 2: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", "", ""


def step3_execution(max_retries=2):
    """
    STEP 3: Code Execution with a retry mechanism

    Mirrors medmars.run() but retries when the generated code raises
    """
    print("\n" + "="*60)
    print("STEP 3: CODE EXECUTION")
    print("="*60)

    if not state.code:
        return "❌ Error: no code available yet", "", ""

    if not state.image_path:
        return "❌ Error: no image provided", "", ""

    out = None
    state.result = None
    error_message = None

    # Retry loop
    for retry_attempt in range(max_retries + 1):
        try:
            # If this is a retry, regenerate code with error feedback
            if retry_attempt > 0:
                print(f"\n{'='*60}")
                print(f"⚠️  Code execution failed. Retrying ({retry_attempt}/{max_retries})...")
                print(f"{'='*60}")

                # Send error + old code back to coder for fixing
                retry_prompt = f"{state.plan}\n\n--- PREVIOUS CODE (FAILED) ---\n```python\n{state.code}\n```\n\n--- ERROR MESSAGE ---\n{error_message}\n\n--- INSTRUCTIONS ---\nThe code above failed with the error shown. Please analyze the error and fix the code. Common issues:\n- KeyError: Check if dict keys exist before accessing\n- AttributeError: Verify object has the method/attribute\n- TypeError: Check data types\n- IndexError: Validate indices\n\nRegenerate the complete execute_command function with the fix."

                print(f"\n🔄 Regenerating code with error feedback...")
                state.code = medmars.code_generator(retry_prompt)
                print(f"💻 Code (Attempt {retry_attempt + 1}):")
                print(state.code[:200] + "...")

            # Execute code - EXACTLY same logic as medmars.run()
            print(f"\n⚙️ Executing code (attempt {retry_attempt + 1})...")
            exec_globals = globals().copy()

            # Create ImagePatch factory with output_dir if specified
            if state.output_dir:
                exec_globals['ImagePatch'] = lambda outputs_dir=str(state.output_dir): ImagePatch(outputs_dir=outputs_dir)
            else:
                exec_globals['ImagePatch'] = ImagePatch

            exec(state.code, exec_globals)

            execute_command = exec_globals.get('execute_command')
            if execute_command is None:
                raise ValueError("execute_command function not found in generated code")

            out = execute_command(state.image_path)
            state.result = out.copy() if hasattr(out, 'copy') else out
            state.execution_output = out

            # Format output as JSON if possible
            output_text = format_output_as_json(out)

            print(f"✅ Code executed successfully on attempt {retry_attempt + 1}!")
            print(f"✅ Execution result: {output_text[:200]}...")

            return f"✅ Step 3: Code executed (attempt {retry_attempt + 1})", output_text, state.code

        except Exception as e:
            error_message = str(e)
            print(f"❌ Error on attempt {retry_attempt + 1}: {error_message}")

            # If this was the last attempt, return error
            if retry_attempt == max_retries:
                state.execution_output = error_message
                state.result = None
                print(f"\n{'='*60}")
                print(f"❌ Code execution failed after {max_retries + 1} attempts.")
                print(f"{'='*60}\n")

                import traceback
                traceback.print_exc()

                return f"❌ Error after {max_retries + 1} attempts: {error_message}", str(e), state.code


def enrich_explanation_with_images(explanation, overlay_images):
    """
    Replace relative image paths in the explanation with base64-embedded images
    and escape <loc_...> tags so Gradio Markdown renders them

    Args:
        explanation: Explanation text from the reporter (may contain markdown images and <loc_...> tags)
        overlay_images: List of overlay image paths

    Returns:
        Markdown text with base64-embedded images and escaped location tags
    """
    import base64
    import re

    # Step 1: Escape <loc_...> tags so Markdown does not strip them
    # Convert <loc_x1_y1_x2_y2> to styled inline code
    def escape_loc_tags(text):
        # Pattern to match <loc_x1_y1_x2_y2> tags
        loc_pattern = r'<loc_(\d+_\d+_\d+_\d+)>'
        # Replace with styled code block that looks like a tag
        def replace_loc(match):
            coords = match.group(1)
            # Use inline code with special styling to make it look like a tag
            return f'`<loc_{coords}>`'

        escaped = re.sub(loc_pattern, replace_loc, text)
        return escaped

    # Apply escaping first
    explanation = escape_loc_tags(explanation)
    print("✅ Escaped location tags in explanation")

    # Step 2: Embed images
    if not overlay_images:
        return explanation

    # Build a mapping of filename -> base64 data URL
    image_map = {}
    for img_path in overlay_images:
        try:
            img_name = Path(img_path).name
            with open(img_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode()
                image_map[img_name] = f"data:image/png;base64,{img_base64}"
            print(f"✅ Loaded image for embedding: {img_name}")
        except Exception as e:
            print(f"⚠️  Warning: could not load image {img_path}: {e}")
            continue

    # Find and replace every markdown image: ![](filename.png) or ![alt](filename.png)
    def replace_image_path(match):
        alt_text = match.group(1)  # Alt text (may be empty)
        filename = match.group(2)  # File name

        print(f"🔎 Matching image: alt=[{alt_text}], filename=[{filename}]")

        # If the alt text is just the filename, derive a display name instead
        if alt_text == filename:
            alt_text = ""

        if filename in image_map:
            data_url = image_map[filename]
            # Use an HTML img tag instead of markdown so it can be styled
            display_name = alt_text if alt_text else filename.replace("overlay_", "").replace("segmentation_", "").replace(".png", "").replace("_", " ").title()
            print(f"✅ Replaced with embedded image: {display_name}")
            return f'<img src="{data_url}" alt="{display_name}" style="max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; padding: 5px;" />'
        else:
            # Leave the reference untouched when the image is missing
            print(f"⚠️  Warning: image {filename} not found in overlay_images")
            return match.group(0)

    # Regex pattern matching markdown images
    # Matches every occurrence, including back-to-back images
    pattern = r'!\[([^\]]*)\]\(([^)]+\.png)\)'
    enriched_explanation = re.sub(pattern, replace_image_path, explanation)

    print(f"\n🔍 Debug: Found {len(re.findall(pattern, explanation))} image references in explanation")

    return enriched_explanation


def step4_generate_answer(question):
    """
    STEP 4: Generate Answer - equivalent to the reporter stage in medmars.run()

    Mirrors this code in medmars.run():
        response = self.reporter(query, out, self.code) if out else {
            "answer": "Error in execution",
            "explanation": str(result),
        }
    """
    print("\n" + "="*60)
    print("STEP 4: GENERATE ANSWER")
    print("="*60)

    if not state.execution_output:
        return "❌ Error: no execution output available yet", "", "", ""

    try:
        # Call reporter - same as medmars.run()
        response = medmars.reporter(question, state.execution_output, state.code) if state.execution_output else {
            "answer": "Error in execution",
            "explanation": str(state.result),
        }

        answer = response.get('answer', 'No answer generated')
        explanation = response.get('explanation') or response.get('reason', 'No explanation available')

        # Collect ALL PNG images in output_dir to embed into the explanation
        # (includes overlay_*.png, segmentation_*.png, etc.)
        overlay_images = []
        if state.output_dir and state.output_dir.exists():
            for img_file in sorted(state.output_dir.glob("*.png")):
                # Skip input image
                if img_file.name != "input_image.jpg":
                    overlay_images.append(str(img_file))
                    print(f"📁 Collected image: {img_file.name}")

        print(f"\n📊 Total images collected: {len(overlay_images)}")
        
        # Debug: Print original explanation
        print(f"\n📄 Original explanation preview:")
        print(explanation[:500])
        
        # Enrich the explanation with embedded images
        explanation_with_images = enrich_explanation_with_images(explanation, overlay_images)
        
        # Debug: Check if replacement happened
        print(f"\n🔄 Explanation changed: {explanation != explanation_with_images}")
        print(f"📏 Original length: {len(explanation)}, New length: {len(explanation_with_images)}")

        print(f"💬 Answer: {answer}")
        print(f"📝 Explanation: {explanation}")
        print(f"🖼️  Embedded {len(overlay_images)} images in explanation")

        return "✅ Step 4: Answer generated", answer, explanation_with_images, ""

    except Exception as e:
        print(f"❌ Error in STEP 4: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", "", "", ""


def step5_collect_images():
    """
    STEP 5: Collect ALL images (not just overlay_*.png)
    """
    print("\n" + "="*60)
    print("STEP 5: COLLECT IMAGES")
    print("="*60)

    try:
        overlay_images = []
        if state.output_dir and state.output_dir.exists():
            # Collect ALL PNG files, not just overlay_*.png
            for img_file in sorted(state.output_dir.glob("*.png")):
                # Skip input image
                if img_file.name != "input_image.jpg":
                    overlay_images.append(str(img_file))
                    print(f"🖼️  Found image: {img_file.name}")

        print(f"✅ Found {len(overlay_images)} images")
        print("="*60)
        print("✅ ALL STEPS COMPLETED!")
        print("="*60 + "\n")

        return "✅ All steps completed!", overlay_images

    except Exception as e:
        print(f"❌ Error in STEP 5: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", []


def run_full_pipeline(image, question):
    """
    Run the whole pipeline - chain all steps together
    Equivalent to medmars.run() but yields after each step
    """
    # Step 1: Planning
    status1, thought, plan, _ = step1_planning(image, question)
    yield status1, "", "", thought, plan, "", "", []

    if "❌" in status1:
        return

    # Step 2: Code Generation
    status2, code, _ = step2_code_generation()
    yield status2, "", "", thought, plan, code, "", []

    if "❌" in status2:
        return

    # Step 3: Execution
    status3, execution_output, _ = step3_execution()
    yield status3, "", "", thought, plan, code, execution_output, []

    if "❌" in status3:
        return

    # Step 4: Generate Answer
    status4, answer, explanation, _ = step4_generate_answer(question)
    yield status4, answer, "", thought, plan, code, execution_output, []

    if "❌" in status4:
        return

    # Step 5: Collect Images
    status5, images = step5_collect_images()
    yield status5, answer, explanation, thought, plan, code, execution_output, images


def create_demo():
    """Build the Gradio interface"""

    with gr.Blocks(title="MedMARS - Medical VQA Assistant") as demo:
        gr.Markdown(
            """
            # 🏥 MedMARS - Medical Visual Question Answering

            **Medical Multi-modal Agent with Reasoning and Search**

            An AI system that analyzes chest X-rays and answers clinical questions.

            ## 🔄 Pipeline (4 steps)
            1. 🧠 **Planning** - Analyze the question and draft a plan
            2. 💻 **Code Generation** - Generate the code to execute
            3. ⚙️ **Execution** - Run the code and collect results
            4. 💬 **Generate Answer** - Produce the final answer
            """
        )

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")

                image_input = gr.Image(
                    type="pil",
                    label="Upload chest X-ray",
                    height=400
                )

                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Example: What abnormalities are present in this chest X-ray?",
                    lines=3
                )

                with gr.Row():
                    run_all_btn = gr.Button("🚀 Run full pipeline", variant="primary", size="lg")
                    gr.ClearButton(
                        components=[image_input, question_input],
                        value="🗑️ Clear"
                    )

                gr.Markdown("""
                    **💡 How to use:**
                    - Upload a chest X-ray
                    - Type your question
                    - Press "🚀 Run full pipeline" to see the result of each step
                """)

                # Examples
                gr.Markdown("### 💡 Examples")
                gr.Examples(
                    examples=[
                        [
                            "src/data/vindr_cxr_vqa/images/0a1aef5326b7b24378c6692f7a454e52.jpg",
                            "What abnormalities are visible in this chest X-ray?"
                        ],
                        [
                            "src/data/vindr_cxr_vqa/images/0a1aef5326b7b24378c6692f7a454e52.jpg",
                            "Is there any pleural effusion?"
                        ],
                        [
                            "src/data/vindr_cxr_vqa/images/0a1aef5326b7b24378c6692f7a454e52.jpg",
                            "Where is the cardiomegaly located?"
                        ]
                    ],
                    inputs=[image_input, question_input],
                    label=None
                )

            # Right column: Output
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")

                # Status indicator
                status_output = gr.Textbox(
                    label="⏳ Status",
                    value="Not started",
                    lines=1,
                    interactive=False
                )

                # Thought (collapsible)
                with gr.Accordion("💭 Thought", open=False):
                    thought_output = gr.Textbox(
                        label="",
                        lines=5,
                        show_label=False
                    )

                # Plan
                plan_output = gr.Textbox(
                    label="📋 Plan",
                    lines=5
                )

                # Code (collapsible)
                with gr.Accordion("💻 Generated Code", open=False):
                    code_output = gr.Code(
                        label="",
                        language="python",
                        lines=10,
                        show_label=False
                    )

                # Code Output (collapsible)
                with gr.Accordion("⚙️ Code Execution Output", open=False):
                    code_result_output = gr.Textbox(
                        label="",
                        lines=5,
                        show_label=False
                    )

                # Answer
                answer_output = gr.Textbox(
                    label="💬 Answer",
                    lines=3
                )

                # Explanation (Markdown format)
                explanation_output = gr.Markdown(
                    label="📝 Detailed explanation"
                )

                # Gallery
                gallery_output = gr.Gallery(
                    label="🖼️ Detected abnormality images",
                    columns=3,
                    height=300,
                    object_fit="contain"
                )

        # Connect button to run full pipeline
        run_all_btn.click(
            fn=run_full_pipeline,
            inputs=[image_input, question_input],
            outputs=[
                status_output,
                answer_output,
                explanation_output,
                thought_output,
                plan_output,
                code_output,
                code_result_output,
                gallery_output
            ]
        )

    return demo


if __name__ == "__main__":
    # Create demo
    print("\n" + "="*60)
    print("🔨 Creating Gradio interface...")
    demo = create_demo()
    print("✅ Interface created successfully!")

    # Launch
    print("\n" + "="*60)
    print("🚀 Launching MedMARS Gradio Interface...")
    print("="*60)
    print("\n📍 Access the app at:")
    print("   - Local: http://localhost:7860")
    print("   - Network: http://0.0.0.0:7860")
    print("\n⚠️  Press Ctrl+C to stop the server\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        quiet=False
    )