"""
MedMARS Gradio Interface
========================
Giao diện web đơn giản để sử dụng MedMARS - Medical Multi-modal Agent with Reasoning and Search

Sử dụng:
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
print("Đang khởi tạo MedMARS...")
medmars = MedMARS()
print("MedMARS đã sẵn sàng!")


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
    Convert object thành JSON-serializable format

    Args:
        obj: Object cần convert (có thể là numpy array, dict, list, etc.)

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
    Format output thành JSON đẹp

    Args:
        output: Output từ code execution

    Returns:
        Formatted string (JSON format đẹp)
    """
    try:
        # Convert thành JSON-serializable object
        serializable = convert_to_serializable(output)

        # Format thành JSON đẹp với indent
        formatted = json.dumps(serializable, indent=2, ensure_ascii=False)

        return formatted

    except Exception:
        # Nếu không thể format thành JSON, dùng pprint
        try:
            import pprint
            return pprint.pformat(output, indent=2, width=80)
        except Exception:
            # Fallback cuối cùng: return string
            return str(output)


def step1_planning(image, question):
    """
    STEP 1: Planning - Tương đương với phần planner trong medmars.run()

    Tương ứng với code trong medmars.run():
        self.thought, self.plan = self.planner(query=query, image_path=image_path)
    """
    print("\n" + "="*60)
    print("STEP 1: PLANNING")
    print("="*60)

    if image is None:
        return "❌ Lỗi: Không có ảnh", "", "", ""

    if not question or question.strip() == "":
        return "❌ Lỗi: Không có câu hỏi", "", "", ""

    try:
        # Create temporary output directory
        state.output_dir = Path("static/temp_gradio")
        state.output_dir.mkdir(parents=True, exist_ok=True)

        # Save image temporarily
        temp_image_path = state.output_dir / "input_image.jpg"
        image.save(str(temp_image_path))
        state.image_path = str(temp_image_path)

        print(f"📝 Câu hỏi: {question}")
        print(f"🖼️  Ảnh đã lưu tại: {state.image_path}")

        # Call planner - same as medmars.run()
        state.thought, state.plan = medmars.planner(query=question, image_path=state.image_path)

        print(f"💭 Thought: {state.thought[:100]}...")
        print(f"📋 Plan: {state.plan[:100]}...")

        return "✅ Bước 1: Đã hoàn thành Planning", state.thought, state.plan, ""

    except Exception as e:
        print(f"❌ Lỗi ở STEP 1: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Lỗi: {str(e)}", "", "", ""


def step2_code_generation():
    """
    STEP 2: Code Generation - Tương đương với phần code_generator trong medmars.run()

    Tương ứng với code trong medmars.run():
        self.code = self.code_generator(self.plan)
    """
    print("\n" + "="*60)
    print("STEP 2: CODE GENERATION")
    print("="*60)

    if not state.plan:
        return "❌ Lỗi: Chưa có plan", "", ""

    try:
        # Call code_generator - same as medmars.run()
        state.code = medmars.code_generator(state.plan)

        print(f"💻 Code: {len(state.code)} chars")
        print("Code preview:")
        print(state.code[:200] + "...")

        return "✅ Bước 2: Đã sinh code", state.code, ""

    except Exception as e:
        print(f"❌ Lỗi ở STEP 2: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Lỗi: {str(e)}", "", ""


def step3_execution():
    """
    STEP 3: Code Execution - Tương đương với phần execution trong medmars.run()

    Tương ứng với code trong medmars.run():
        exec_globals = globals().copy()
        if output_dir:
            exec_globals['ImagePatch'] = lambda outputs_dir=output_dir: ImagePatch(outputs_dir=outputs_dir)
        else:
            exec_globals['ImagePatch'] = ImagePatch
        exec(self.code, exec_globals)
        execute_command = exec_globals.get('execute_command')
        out = execute_command(image_path)
    """
    print("\n" + "="*60)
    print("STEP 3: CODE EXECUTION")
    print("="*60)

    if not state.code:
        return "❌ Lỗi: Chưa có code", "", ""

    if not state.image_path:
        return "❌ Lỗi: Không có ảnh", "", ""

    try:
        # Execute code - EXACTLY same logic as medmars.run()
        exec_globals = globals().copy()

        # Create ImagePatch factory with output_dir if specified
        if state.output_dir:
            exec_globals['ImagePatch'] = lambda outputs_dir=str(state.output_dir): ImagePatch(outputs_dir=outputs_dir)
        else:
            exec_globals['ImagePatch'] = ImagePatch

        exec(state.code, exec_globals)
        state.result = None

        try:
            execute_command = exec_globals.get('execute_command')
            if execute_command is None:
                raise ValueError("execute_command function not found in generated code")

            out = execute_command(state.image_path)
            state.result = out.copy() if hasattr(out, 'copy') else out
            state.execution_output = out

            # Format output as JSON if possible
            output_text = format_output_as_json(out)

            print(f"✅ Execution result: {output_text[:200]}...")

        except Exception as e:
            out = str(e)
            state.execution_output = out
            state.result = None
            output_text = str(e)
            print(f"❌ Execution error: {out}")

        return "✅ Bước 3: Đã thực thi code", output_text, ""

    except Exception as e:
        print(f"❌ Lỗi ở STEP 3: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Lỗi: {str(e)}", "", ""


def enrich_explanation_with_images(explanation, overlay_images):
    """
    Thêm overlay images vào explanation dạng markdown

    Args:
        explanation: Explanation text từ reporter
        overlay_images: List các đường dẫn overlay images

    Returns:
        Markdown text với embedded images
    """
    if not overlay_images:
        return explanation

    # Thêm section cho overlay images
    image_section = "\n\n---\n\n### 🖼️ Hình ảnh minh họa\n\n"

    for i, img_path in enumerate(overlay_images, 1):
        # Lấy tên file
        img_name = Path(img_path).stem.replace("overlay_", "").replace("_", " ").title()
        # Thêm markdown image syntax
        image_section += f"**{i}. {img_name}**\n\n"
        image_section += f"![{img_name}]({img_path})\n\n"

    return explanation + image_section


def step4_generate_answer(question):
    """
    STEP 4: Generate Answer - Tương đương với phần reporter trong medmars.run()

    Tương ứng với code trong medmars.run():
        response = self.reporter(query, out, self.code) if out else {
            "answer": "Error in execution",
            "explanation": str(result),
        }
    """
    print("\n" + "="*60)
    print("STEP 4: GENERATE ANSWER")
    print("="*60)

    if not state.execution_output:
        return "❌ Lỗi: Chưa có execution output", "", "", ""

    try:
        # Call reporter - same as medmars.run()
        response = medmars.reporter(question, state.execution_output, state.code) if state.execution_output else {
            "answer": "Error in execution",
            "explanation": str(state.result),
        }

        answer = response.get('answer', 'No answer generated')
        explanation = response.get('explanation') or response.get('reason', 'No explanation available')

        # Collect overlay images để nhúng vào explanation
        overlay_images = []
        if state.output_dir and state.output_dir.exists():
            for img_file in sorted(state.output_dir.glob("overlay_*.png")):
                overlay_images.append(str(img_file))

        # Enrich explanation với embedded images
        explanation_with_images = enrich_explanation_with_images(explanation, overlay_images)

        print(f"💬 Answer: {answer}")
        print(f"📝 Explanation: {explanation[:100]}...")
        print(f"🖼️  Embedded {len(overlay_images)} images in explanation")

        return "✅ Bước 4: Đã tạo câu trả lời", answer, explanation_with_images, ""

    except Exception as e:
        print(f"❌ Lỗi ở STEP 4: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Lỗi: {str(e)}", "", "", ""


def step5_collect_images():
    """
    STEP 5: Collect overlay images
    """
    print("\n" + "="*60)
    print("STEP 5: COLLECT IMAGES")
    print("="*60)

    try:
        overlay_images = []
        if state.output_dir and state.output_dir.exists():
            for img_file in sorted(state.output_dir.glob("overlay_*.png")):
                overlay_images.append(str(img_file))
                print(f"🖼️  Found overlay: {img_file.name}")

        print(f"✅ Tìm thấy {len(overlay_images)} overlay images")
        print("="*60)
        print("✅ HOÀN THÀNH TẤT CẢ CÁC BƯỚC!")
        print("="*60 + "\n")

        return "✅ Hoàn thành tất cả!", overlay_images

    except Exception as e:
        print(f"❌ Lỗi ở STEP 5: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Lỗi: {str(e)}", []


def run_full_pipeline(image, question):
    """
    Chạy toàn bộ pipeline - chain tất cả các steps lại với nhau
    Tương đương với medmars.run() nhưng yield từng bước
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
    """Tạo Gradio interface"""

    with gr.Blocks(title="MedMARS - Medical VQA Assistant") as demo:
        gr.Markdown(
            """
            # 🏥 MedMARS - Medical Visual Question Answering

            **Medical Multi-modal Agent with Reasoning and Search**

            Hệ thống AI phân tích ảnh X-quang ngực và trả lời câu hỏi y khoa.

            ## 🔄 Quy trình xử lý (4 bước)
            1. 🧠 **Planning** - Phân tích câu hỏi và tạo kế hoạch
            2. 💻 **Code Generation** - Sinh code thực thi
            3. ⚙️ **Execution** - Chạy code và lấy kết quả
            4. 💬 **Generate Answer** - Tạo câu trả lời
            """
        )

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")

                image_input = gr.Image(
                    type="pil",
                    label="Upload ảnh X-quang ngực",
                    height=400
                )

                question_input = gr.Textbox(
                    label="Câu hỏi",
                    placeholder="Ví dụ: What abnormalities are present in this chest X-ray?",
                    lines=3
                )

                with gr.Row():
                    run_all_btn = gr.Button("🚀 Chạy toàn bộ", variant="primary", size="lg")
                    gr.ClearButton(
                        components=[image_input, question_input],
                        value="🗑️ Xóa"
                    )

                gr.Markdown("""
                    **💡 Cách sử dụng:**
                    - Upload ảnh X-quang
                    - Nhập câu hỏi
                    - Nhấn "🚀 Chạy toàn bộ" để xem kết quả từng bước
                """)

                # Examples
                gr.Markdown("### 💡 Ví dụ")
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
                gr.Markdown("### 📊 Kết quả")

                # Status indicator
                status_output = gr.Textbox(
                    label="⏳ Trạng thái",
                    value="Chưa bắt đầu",
                    lines=1,
                    interactive=False
                )

                # Thought (collapsible)
                with gr.Accordion("💭 Thought (Suy nghĩ)", open=False):
                    thought_output = gr.Textbox(
                        label="",
                        lines=5,
                        show_label=False
                    )

                # Plan
                plan_output = gr.Textbox(
                    label="📋 Plan (Kế hoạch)",
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
                    label="💬 Câu trả lời",
                    lines=3
                )

                # Explanation (Markdown format)
                explanation_output = gr.Markdown(
                    label="📝 Giải thích chi tiết"
                )

                # Gallery
                gallery_output = gr.Gallery(
                    label="🖼️ Hình ảnh phát hiện bệnh",
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
    print("\n📍 Truy cập app tại:")
    print("   - Local: http://localhost:7860")
    print("   - Network: http://0.0.0.0:7860")
    print("\n⚠️  Nhấn Ctrl+C để dừng server\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        quiet=False
    )