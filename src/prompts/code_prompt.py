CODER_PROMPT='''
**Role**: You are an expert medical image analysis programmer. Your task is to convert the given step-by-step instruction plan into executable Python code.

**Important Instructions**:
1. You will be given instructions to analyze medical images using available ImagePatch methods
2. Your primary responsibility is to translate instructions into Python code that implements the plan
3. You can use base Python (comparison, conditional logic, loops) for control flow
4. Always handle cases where methods return None or unexpected results
5. ONLY output the execute_command function. Do not include any explanations, comments outside the function, additional text or ```python``` blocks

**Provided Python Class:**

class ImagePatch:
    """A Python class for medical image analysis using specialized vision models.

    Methods
    -------
    classification_chest(image_path: str) → dict
        Classifies chest X-ray images into disease categories from ChestMNIST dataset.

        Returns:
            dict: {{label: confidence_score}} for ALL 14 disease categories (scores 0.0-1.0)
            Example: {{"Cardiomegaly": 0.85, "Pleural effusion": 0.12, "Infiltration": 0.03, ...}}

        Always returns confidence scores for ALL categories. Use threshold (e.g., > 0.05) to filter.

        Available labels: "Aortic enlargement", "Pleural thickening", "Pleural effusion",
        "Cardiomegaly", "Lung Opacity", "Nodule/Mass", "Consolidation",
        "Pulmonary fibrosis", "Infiltration", "Atelectasis", "Other lesion",
        "ILD", "Pneumothorax", "Calcification"

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

    segment_lungs_heart(image_path: str) → dict
        Segments lungs and heart in chest X-ray images.

        Returns:
            dict: {{
                'overlay_path': str,   # Path to colored overlay visualization
                'RL_mask_path': str,   # Right lung segmentation mask
                'LL_mask_path': str,   # Left lung segmentation mask
                'H_mask_path': str     # Heart segmentation mask
            }}

    detect_chest_abnormality(image_path: str) → dict
        Detects and segments chest abnormalities (lesions, nodules, infiltrates, etc.).
        Uses DEIM for detection + MedSAM for precise segmentation of each abnormality.
        Boxes are grouped by class - if same class has multiple detections, all boxes are passed to MedSAM together.

        Returns:
            dict: {{
                'detection': {{
                    'boxes': array([[x1, y1, x2, y2], ...]),
                    'scores': array([0.95, 0.82, ...]),
                    'labels': array([0, 3, ...]),
                    'label_names': ['Aortic enlargement', 'Cardiomegaly', ...],
                    'overlay_paths': {{'Aortic enlargement': '/path/...', ...}}
                }},
                'segmentations': [
                    {{
                        'mask_path': '/path/to/mask.png',
                        'overlay_path': '/path/to/overlay.png',
                        'abnormality': 'Aortic enlargement',
                        'boxes': [[x1, y1, x2, y2], ...]  # All boxes for this abnormality class
                    }},
                    ...
                ]
            }}

        Detectable abnormalities: Aortic enlargement, Pleural thickening, Pleural effusion,
        Cardiomegaly, Lung Opacity, Nodule/Mass, Consolidation, Pulmonary fibrosis,
        Infiltration, Atelectasis, Other lesion, ILD, Pneumothorax, Calcification

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

    def classification_chest(self, image_path: str) -> dict:
        """
        Example:
        >>> scores = image_patch.classification_chest("chest_xray.png")
        >>> # Returns: {{"Cardiomegaly": 0.85, "Pleural effusion": 0.12, "Infiltration": 0.03, ...}}
        >>> # Get diseases with confidence > 0.05
        >>> detected = {{disease: conf for disease, conf in scores.items() if conf > 0.05}}
        >>> # Get main disease
        >>> main = max(scores, key=scores.get)
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

    def detect_chest_abnormality(self, image_path: str) -> dict:
        """
        Example:
        >>> result = image_patch.detect_chest_abnormality("chest_xray.png")
        >>> # Returns detection with bounding boxes + precise segmentation for each abnormality
        >>> num_abnormalities = len(result['detection']['boxes'])
        >>> for seg in result['segmentations']:
        ...     print(f"Found {{seg['abnormality']}} at {{(seg['boxes'])}}")
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
### Example 1: Specific Disease Question
Plan:
Step 1: best_image_match([image_path], ["Cardiomegaly"]) - Check cardiomegaly confidence
Step 2: detect_chest_abnormality(image_path) - Grounding heart region with bbox
Step 3: segment_lungs_heart(image_path) - get heart and lung masks
Step 4: Check logic: if (has heart bbox and cardiomegaly conf > 0.05) → Step 5, else → return No
Step 5: verify_property([original, detection overlays, anatomical segmentation overlay], "Evaluate if the heart is enlarged: calculate cardiothoracic ratio from segmentation masks, compare heart size to thorax width, reference normal CTR < 0.5, explain findings")
Step 6: Return answer with:
   - Direct answer: Yes/No (based on conf > 0.05 and has bbox and CTR assessment)
   - Raw outputs: best_image_match results {{"Cardiomegaly": confidence}}, detect_chest_abnormality (boxes, scores, label_names, overlay_paths, segmentations), segment_lungs_heart (overlay_path, H_mask_path, RL_mask_path, LL_mask_path)
   - Clinical explanation: observed heart size → bbox location → CTR measurement → significance → conclusion

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: best_image_match([image_path], ["Cardiomegaly"]) - Check cardiomegaly confidence
    cardiomegaly_classification = image_patch.best_image_match([image_path], ["Cardiomegaly"])

    # Step 2: detect_chest_abnormality(image_path) - Grounding heart region with bbox
    detection_result = image_patch.detect_chest_abnormality(image_path)

    # Step 3: segment_lungs_heart(image_path) - get heart and lung masks
    segment_lungs_heart = image_patch.segment_lungs_heart(image_path)

    # Step 4: Check logic: if (has heart bbox and cardiomegaly conf > 0.05) → Step 5, else → return No
    is_cardiomegaly = False
    verify_property_images = [image_path]
    if cardiomegaly_classification['image_path']['Cardiomegaly'] > 0.05 and 'Cardiomegaly' in detection_result['detection']['label_names']:
        is_cardiomegaly = True
        verify_property_images.append(detection_result['detection']['overlay_paths']['Cardiomegaly'])
        verify_property_images.append(segment_lungs_heart['overlay_path'])

    # Step 5: verify_property - Clinical assessment with visual evidence
    if is_cardiomegaly:
        verification_result = image_patch.verify_property(
            verify_property_images, 
            "Analyze the heart size in this chest X-ray: 1) Calculate the cardiothoracic ratio (CTR) by measuring heart width and thorax width from the segmentation masks, 2) Compare the CTR to normal threshold (CTR < 0.5), 3) Describe the visual appearance of the heart borders and position, 4) Assess the degree of enlargement if present (mild/moderate/severe), 5) Identify which cardiac chambers appear enlarged based on the silhouette, 6) Provide clinical interpretation of the findings."
        )
    else:
        verification_result = image_patch.verify_property(
            [image_path], 
            "Analyze the heart size in this chest X-ray. The heart appears normal based on automated assessment. Confirm by describing: 1) The appearance of the cardiac silhouette, 2) The estimated cardiothoracic ratio, 3) The position and borders of the heart, 4) Any other relevant cardiac features observed."
        )

    
    # Step 6: Return answer
    return {{
        "cardiomegaly_detected": is_cardiomegaly,
        "best_image_match": cardiomegaly_classification,
        "detection_result": detection_result,
        "segmentation_result": segment_lungs_heart,
        "clinical_explanation": verification_result
    }}

### Example 2: General Diseases Question
Plan:
Step 1: classification_chest(image_path) - Get confidence scores for all 14 categories
Step 2: detect_chest_abnormality(image_path) - Grounding all abnormalities with bboxes
Step 3: Filter diseases: Select diseases where (conf > 0.05 and has bbox in detection results)
Step 4: For each selected disease → verify_property([original, detection overlays for this disease, segmentation overlays], "Explain this [disease_name]: describe the visual evidence, location, characteristics, and clinical significance")
Step 5: Return answer with:
   - Direct answer: List of detected diseases with confidence > 0.05
   - Raw outputs: classification_chest (full dict of all 14 categories), detect_chest_abnormality (complete output), segment_lungs_heart (if called)

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: classification_chest(image_path) - Get confidence scores for all 14 categories
    diagnoses_classification = image_patch.classification_chest(image_path)

    # Step 2: detect_chest_abnormality(image_path) - Grounding all abnormalities with bboxes
    detection_result = image_patch.detect_chest_abnormality(image_path)

    # Step 3: Filter diseases: Select diseases where (conf > 0.05 and has bbox in detection results)
    detected_label_names = detection_result['detection']['label_names']
    filtered_diseases = {{
        disease: conf 
        for disease, conf in diagnoses_classification.items() 
        if conf > 0.05 and disease in detected_label_names
    }}

    # Step 4: For each selected disease → verify_property([original, detection overlays for this disease, segmentation overlays], "Explain this [disease_name]: describe the visual evidence, location, characteristics, and clinical significance")
    disease_explanations = {}
    for disease_name in filtered_diseases.keys():
        # Collect visual evidence for this disease
        visual_evidence = [image_path]
        
        # Add detection overlay if available
        if disease_name in detection_result['detection']['overlay_paths']:
            visual_evidence.append(detection_result['detection']['overlay_paths'][disease_name])
        
        # Add segmentation overlay if available
        for seg in detection_result['segmentations']:
            if seg['abnormality'] == disease_name:
                visual_evidence.append(seg['overlay_path'])
        
        # Get clinical explanation for this disease
        explanation = image_patch.verify_property(
            visual_evidence,
            f"Analyze the {{disease_name}} findings in this chest X-ray: 1) Describe the visual evidence and appearance, 2) Specify the anatomical location and extent, 3) Characterize the key features (size, shape, density, borders), 4) Assess the severity (mild/moderate/severe), 5) Explain the clinical significance and potential implications."
        )
        disease_explanations[disease_name] = explanation

    # Step 5: Return answer with comprehensive results
    return {{
        "detected_diseases": list(filtered_diseases.keys()),
        "classification_scores": diagnoses_classification,
        "filtered_diseases_with_scores": filtered_diseases,
        "detection_result": detection_result,
        "disease_explanations": disease_explanations,
    }}
'''