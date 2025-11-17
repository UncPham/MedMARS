import os
import sys
import csv
from pathlib import Path
from datetime import datetime
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.medmars import MedMARS
from src.image_patch import ImagePatch

# Configuration
NUM_QUESTIONS_TO_TEST = None  # Set to None to test all questions, or a number like 10 to test first 10
CSV_PATH = "src/data/vqa_rad/vqa_rad.csv"
IMAGES_DIR = "src/data/vqa_rad/images"

# Generate timestamp for this test run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_BASE_DIR = f"logs/{TIMESTAMP}"

def is_yes_no_question(answer: str) -> bool:
    """Check if the answer is yes/no type"""
    answer_lower = answer.lower().strip()
    return answer_lower in ['yes', 'no']

def calculate_accuracy(predicted: str, ground_truth: str) -> float:
    """Calculate accuracy for yes/no questions.
    An answer is counted correct if:
    - Exact match (case-insensitive), or
    - The first word of the predicted answer matches the ground truth (e.g., "Yes, there is..." vs "Yes").
    """
    pred_lower = predicted.lower().strip()
    gt_lower = ground_truth.lower().strip()

    if pred_lower == gt_lower:
        return 1.0

    # Check first token match for answers like "Yes, ..." or "No. ..."
    first_token = pred_lower.split()[0] if pred_lower else ""
    if first_token in ["yes", "no"] and first_token == gt_lower:
        return 1.0

    return 0.0

def calculate_recall(predicted: str, ground_truth: str) -> float:
    """Calculate recall score for open-ended questions"""
    # Simple word-based recall
    pred_words = set(predicted.lower().split())
    gt_words = set(ground_truth.lower().split())

    if len(gt_words) == 0:
        return 0.0

    # Count how many ground truth words appear in prediction
    matching_words = pred_words.intersection(gt_words)
    recall = len(matching_words) / len(gt_words)
    return recall

def save_markdown_report(output_dir: str, question_data: dict):
    """Save detailed markdown report for each question"""
    markdown_path = os.path.join(output_dir, "report.md")

    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(f"# Medical Question Analysis Report\n\n")
        f.write(f"## Question\n{question_data['question']}\n\n")
        f.write(f"## Image\n`{question_data['image_path']}`\n\n")

        f.write(f"## Thought\n```\n{question_data['thought']}\n```\n\n")
        f.write(f"## Plan\n```\n{question_data['plan']}\n```\n\n")
        f.write(f"## Generated Code\n```python\n{question_data['code']}\n```\n\n")
        f.write(f"## Code Output\n```\n{question_data['output']}\n```\n\n")

        f.write(f"## Final Answer\n{question_data['answer']}\n\n")
        f.write(f"## Explanation\n{question_data['explanation']}\n\n")

        f.write(f"## Ground Truth\n{question_data['ground_truth']}\n\n")
        f.write(f"## Score\n{question_data['score']:.4f}\n\n")
        f.write(f"## Score Type\n{question_data['score_type']}\n\n")

def setup_logging(test_run_dir):
    """Setup logging to file and console"""
    log_file = os.path.join(test_run_dir, "test.log")

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def test_medmars():
    """Main test function"""
    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_BASE_DIR)
    logger.info(f"Starting test run at {TIMESTAMP}")
    logger.info(f"Test results will be saved to: {OUTPUT_BASE_DIR}")

    # Read CSV file
    questions = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    # Limit number of questions if specified
    if NUM_QUESTIONS_TO_TEST is not None:
        questions = questions[:NUM_QUESTIONS_TO_TEST]

    logger.info(f"Testing {len(questions)} questions...")

    # Initialize MedMARS
    logger.info("Initializing MedMARS...")
    medmars = MedMARS()

    # Track scores
    yes_no_scores = []
    open_ended_scores = []
    all_results = []

    for i, q in enumerate(questions):
        logger.info("="*80)
        logger.info(f"Question {i+1}/{len(questions)}")
        logger.info("="*80)

        file_name = q['file_name']
        question = q['question']
        ground_truth = q['answer']

        image_path = os.path.join(IMAGES_DIR, file_name)

        # Create output directory for this question
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"vqa_{i}")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Image: {file_name}")
        logger.info(f"Question: {question}")
        logger.info(f"Ground Truth: {ground_truth}")

        try:
            # Run MedMARS with output_dir for this question
            thought, plan, code, output, result, response = medmars.run(
                query=question,
                image=image_path,
                output_dir=output_dir
            )

            # Extract answer from response
            answer = response.get('answer', '')
            explanation = response.get('explanation', '')

            # Calculate score
            if is_yes_no_question(ground_truth):
                score = calculate_accuracy(answer, ground_truth)
                score_type = "accuracy"
                yes_no_scores.append(score)
            else:
                score = calculate_recall(answer, ground_truth)
                score_type = "recall"
                open_ended_scores.append(score)

            logger.info(f"Predicted Answer: {answer}")
            logger.info(f"Score ({score_type}): {score:.4f}")

            # Prepare data for markdown report
            question_data = {
                'question': question,
                'image_path': image_path,
                'thought': thought if thought else "N/A",
                'plan': plan if plan else "N/A",
                'code': code if code else "N/A",
                'output': str(output) if output else "N/A",
                'answer': answer,
                'explanation': explanation,
                'ground_truth': ground_truth,
                'score': score,
                'score_type': score_type
            }

            # Save markdown report
            save_markdown_report(output_dir, question_data)

            all_results.append({
                'question_id': i,
                'file_name': file_name,
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': answer,
                'score': score,
                'score_type': score_type
            })

        except Exception as e:
            logger.error(f"Error processing question {i+1}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # Save error report
            error_data = {
                'question': question,
                'image_path': image_path,
                'thought': "ERROR",
                'plan': "ERROR",
                'code': "ERROR",
                'output': str(e),
                'answer': "ERROR",
                'explanation': str(e),
                'ground_truth': ground_truth,
                'score': 0.0,
                'score_type': "error"
            }
            save_markdown_report(output_dir, error_data)

            all_results.append({
                'question_id': i,
                'file_name': file_name,
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': "ERROR",
                'score': 0.0,
                'score_type': "error"
            })

    # Calculate and print overall statistics
    logger.info("="*80)
    logger.info("OVERALL RESULTS")
    logger.info("="*80)

    if yes_no_scores:
        avg_accuracy = sum(yes_no_scores) / len(yes_no_scores)
        logger.info(f"Yes/No Questions - Average Accuracy: {avg_accuracy:.4f} ({len(yes_no_scores)} questions)")

    if open_ended_scores:
        avg_recall = sum(open_ended_scores) / len(open_ended_scores)
        logger.info(f"Open-Ended Questions - Average Recall: {avg_recall:.4f} ({len(open_ended_scores)} questions)")

    # Save summary CSV
    summary_path = os.path.join(OUTPUT_BASE_DIR, "test_summary.csv")
    with open(summary_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question_id', 'file_name', 'question', 'ground_truth', 'predicted_answer', 'score', 'score_type'])
        writer.writeheader()
        writer.writerows(all_results)

    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info(f"Individual reports saved to: {OUTPUT_BASE_DIR}/vqa_*/report.md")
    logger.info(f"Test log saved to: {OUTPUT_BASE_DIR}/test.log")
    logger.info(f"Test run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    test_medmars()
