CODER_PROMPT='''
**Role**: You are an expert medical image analysis programmer. Your task is to convert the given step-by-step instruction plan into executable Python code.

**Important Instructions**:
1. You will be given instructions to analyze medical images using available ImagePatch methods
2. Your primary responsibility is to translate instructions into Python code that implements the plan
3. You can use base Python (comparison, conditional logic, loops) for control flow
4. Always handle cases where methods return None or unexpected results
5. ONLY output the execute_command function. Do not include any explanations, comments outside the function, or additional text

**Provided Python Class:**

class ImagePatch:
    """A Python class for medical image analysis using specialized vision models.

    Methods
    -------
    classification_chest(image_path: str) → str or None
        Classifies chest X-ray images into disease categories from ChestMNIST dataset.

        Returns:
            str: Disease label if confidence > 0.4
            None: If no disease detected with sufficient confidence

        Available labels: "atelectasis", "cardiomegaly", "effusion", "infiltration",
        "mass", "nodule", "pneumonia", "pneumothorax", "consolidation", "edema",
        "emphysema", "fibrosis", "pleural", "hernia"

        Example: "pneumonia" or None

    best_image_match(images_path: list[str], labels: list[str]) → dict
        Matches multiple images to multiple labels with confidence scores.

        Returns:
            dict: {{
                'image1.jpg': {{
                    'label1': 99.99,  # confidence score (0-100)
                    'label2': 0.01,
                    'label3': 0.0,
                    ...
                }},
                'image2.jpg': {{...}},
                ...
            }}

        Example:
            Input: images=['img1.jpg'], labels=['chest X-ray', 'brain MRI']
            Output: {{'img1.jpg': {{'chest X-ray': 99.5, 'brain MRI': 0.5}}}}

    segment_lungs_heart(image_path: str) → dict
        Segments lungs and heart in chest X-ray images.

        Returns:
            dict: {{
                'overlay_path': str,   # Path to colored overlay visualization
                'RL_mask_path': str,   # Right lung segmentation mask
                'LL_mask_path': str,   # Left lung segmentation mask
                'H_mask_path': str     # Heart segmentation mask
            }}

        Example: {{
            'overlay_path': '/path/to/chest_overlay.png',
            'RL_mask_path': '/path/to/right_lung.png',
            'LL_mask_path': '/path/to/left_lung.png',
            'H_mask_path': '/path/to/heart.png'
        }}

    verify_property(list_image_path: list[str], query: str) → str
        Verifies specific visual properties or answers detailed questions about images.
        Can analyze multiple images including processed outputs from vision models.
        Returns natural language explanation/answer.

        IMPORTANT: Always pass image paths as a LIST
        - Single image: [image_path]
        - Multiple images: [image_path, overlay_path, mask_path]

        BEST PRACTICE: Include processed images (overlays, segmentation masks) from
        vision models for more accurate and detailed analysis by the explainer.

        Use for:
        - Property verification: "is the lesion raised?", "does it have irregular borders?"
        - Descriptive questions: "what is the texture?", "what color is the affected area?"
        - Complex assessments: "are there signs of inflammation?"
        - Comparative queries: "is this larger than normal?"
        - Multi-image analysis: Analyze original + processed images together
    """

    def classification_chest(self, image_path: str) -> str:
        """
        Example:
        >>> diagnosis = image_patch.classification_chest("chest_xray.png")
        >>> # Returns: "pneumonia" or None if confidence < 0.4
        """
        pass

    def classification_organa(self, image_path: str) -> str:
        """
        Example:
        >>> organ = image_patch.classification_organa("organ_scan.png")
        >>> # Returns: organ label or None if confidence < 0.4
        """
        pass

    def best_image_match(self, images_path: list[str], labels: list[str]) -> dict:
        """
        Example:
        >>> results = image_patch.best_image_match(
        ...     ["img1.png", "img2.png"],
        ...     ["pneumonia", "normal", "chest X-ray"]
        ... )
        >>> # Returns: {{
        >>> #   "img1.png": {{"pneumonia": 92.5, "normal": 5.2, "chest X-ray": 2.3}},
        >>> #   "img2.png": {{"pneumonia": 3.1, "normal": 95.8, "chest X-ray": 1.1}}
        >>> # }}
        >>> # Each image gets confidence scores (0-100) for ALL provided labels
        """
        pass

    def segment_lungs_heart(self, image_path: str) -> dict:
        """
        Example:
        >>> result = image_patch.segment_lungs_heart("chest_xray.png")
        >>> # Returns: {{
        >>> #   'overlay_path': '/path/to/overlay.png',
        >>> #   'RL_mask_path': '/path/to/right_lung.png',
        >>> #   'LL_mask_path': '/path/to/left_lung.png',
        >>> #   'H_mask_path': '/path/to/heart.png'
        >>> # }}
        """
        pass

    def verify_property(self, list_image_path: list[str], query: str) -> str:
        """
        Example:
        >>> answer = image_patch.verify_property(
        ...     [image_path],
        ...     "Are there signs of inflammation?"
        ... )
        >>> # Returns: "Yes, there are signs of acute inflammation including..."
        """
        pass

### Examples
{example}

**CRITICAL OUTPUT REQUIREMENTS**:
1. Output ONLY the execute_command function code
2. Do NOT include explanations, markdown formatting, or additional text
3. The function must start with: def execute_command(image_path):
4. Do NOT wrap the code in ```python ``` blocks
5. The code should be ready to execute directly

**Expected format**:
def execute_command(image_path):
    # your implementation here
    pass
'''

EXAMPLES_CODER = '''
### Example 1: Pneumonia Detection
Plan:
Step 1: Use segment_lungs_heart to visualize affected regions
Step 2: Use classification_chest(image_path) to classify the chest X-ray
Step 3: If result is "pneumonia", use verify_property with segmentation images for detailed analysis
Step 4: Return comprehensive results

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Segment lungs and heart for visualization
    segmentation = image_patch.segment_lungs_heart(image_path)

    # Step 2: Classify chest X-ray
    diagnosis = image_patch.classification_chest(image_path)

    # Step 3: Detailed analysis if pneumonia detected
    # Use segmentation overlay and original image for explainer
    detailed_analysis = None
    if diagnosis == "pneumonia":
        detailed_analysis = image_patch.verify_property(
            [image_path, segmentation['overlay_path']],
            "Describe the pneumonia signs in this chest X-ray including location, extent, and pattern. Pay attention to the segmented lung regions."
        )

    # Step 4: Return results
    return {{
        "pneumonia_detected": diagnosis == "pneumonia",
        "diagnosis": diagnosis,
        "detailed_analysis": detailed_analysis,
        "segmentation_paths": segmentation,
    }}

### Example 2: Medical Knowledge Question
Plan:
Step 1: Return definition of pneumonia

A:
def execute_command(image_path=None):
    return {{
        "question": "What is pneumonia?",
        "answer": "Pneumonia is an infection of the lung tissue causing inflammation of the air sacs (alveoli), which fill with fluid or pus. Common symptoms include cough, fever, difficulty breathing, and chest pain. It can be caused by bacteria, viruses, or fungi.",
        "type": "medical_knowledge"
    }}

### Example 3: Heart Enlargement Assessment
Plan:
Step 1: Use segment_lungs_heart to visualize heart
Step 2: Use classification_chest to check for cardiomegaly
Step 3: Use verify_property for detailed assessment
Step 4: Return yes/no answer with supporting evidence

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Segment heart
    segmentation = image_patch.segment_lungs_heart(image_path)

    # Step 2: Classify for cardiomegaly
    diagnosis = image_patch.classification_chest(image_path)

    # Step 3: Detailed assessment with segmentation overlay
    assessment = image_patch.verify_property(
        [image_path, segmentation['overlay_path'], segmentation['H_mask_path']],
        "Assess the heart size and calculate the cardiothoracic ratio. Is the heart enlarged? Analyze the heart segmentation provided."
    )

    # Step 4: Return comprehensive answer
    is_enlarged = diagnosis == "cardiomegaly"

    return {{
        "heart_enlarged": is_enlarged,
        "classification": diagnosis,
        "heart_mask": segmentation['H_mask_path'],
        "overlay": segmentation['overlay_path'],
        "detailed_assessment": assessment,
        "recommendation": "Cardiology consultation recommended" if is_enlarged else "Heart size appears normal"
    }}

### Example 4: Comprehensive Chest X-Ray Analysis
Plan:
Step 1: Use segment_lungs_heart for anatomical segmentation
Step 2: Use classification_chest to identify primary findings
Step 3: Use verify_property for comprehensive analysis
Step 4: Return complete report

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Anatomical segmentation
    segmentation = image_patch.segment_lungs_heart(image_path)

    # Step 2: Classify disease
    disease = image_patch.classification_chest(image_path)

    # Step 3: Comprehensive analysis with all segmentation images
    comprehensive_analysis = image_patch.verify_property(
        [image_path, segmentation['overlay_path']],
        "Provide a comprehensive analysis of this chest X-ray including: anatomical structures visible, any abnormalities, their location and characteristics, and clinical significance. Use the segmentation overlay to identify specific regions."
    )

    # Step 4: Return complete report
    return {{
        "primary_finding": disease if disease else "No specific disease classified",
        "anatomical_segmentation": {{
            "overlay": segmentation['overlay_path'],
            "right_lung": segmentation['RL_mask_path'],
            "left_lung": segmentation['LL_mask_path'],
            "heart": segmentation['H_mask_path']
        }},
        "comprehensive_analysis": comprehensive_analysis,
        "has_abnormality": disease is not None,
        "recommendation": "Medical review recommended" if disease else "Appears normal"
    }}
'''