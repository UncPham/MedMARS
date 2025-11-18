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
    classification_chest(image_path: str) → list[str] or None
        Classifies chest X-ray images into disease categories from ChestMNIST dataset.

        Returns:
            list[str]: List of all detected disease labels with confidence > 0.4
            None: If no disease detected with sufficient confidence

        Can detect multiple conditions simultaneously in the same image.

        Available labels: "Aortic enlargement", "Pleural thickening", "Pleural effusion",
        "Cardiomegaly", "Lung Opacity", "Nodule/Mass", "Consolidation",
        "Pulmonary fibrosis", "Infiltration", "Atelectasis", "Other lesion",
        "ILD", "Pneumothorax", "Calcification"

        Example: ["Cardiomegaly", "Pleural effusion"] or ["Infiltration"] or None

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

    detect_chest_abnormality(image_path: str) → dict
        Detects and segments chest abnormalities (lesions, nodules, infiltrates, etc.).
        Uses DEIM for detection + MedSAM for precise segmentation of each abnormality.

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
                        'box': [x1, y1, x2, y2]
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

    def classification_chest(self, image_path: str) -> list:
        """
        Example:
        >>> diagnoses = image_patch.classification_chest("chest_xray.png")
        >>> # Returns: ["Infiltration", "Pleural effusion"] or ["Cardiomegaly"] or None if confidence < 0.4
        >>> # Can detect multiple conditions in same image
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

    def detect_chest_abnormality(self, image_path: str) -> dict:
        """
        Detects and segments chest abnormalities using DEIM + MedSAM models.

        Returns:
            dict: {{
                'detection': {{
                    'boxes': numpy array of bounding boxes [x1, y1, x2, y2],
                    'scores': confidence scores (0-1),
                    'labels': numeric labels,
                    'label_names': list of abnormality names,
                    'overlay_paths': {{class_name: overlay_image_path}}
                }},
                'segmentations': [
                    {{
                        'mask_path': str,
                        'overlay_path': str,
                        'abnormality': str,
                        'box': [x1, y1, x2, y2]
                    }},
                    ...
                ]
            }}

        Detected categories: Aortic enlargement, Pleural thickening, Pleural effusion,
        Cardiomegaly, Lung Opacity, Nodule/Mass, Consolidation, Pulmonary fibrosis,
        Infiltration, Atelectasis, Other lesion, ILD, Pneumothorax, Calcification

        Example:
        >>> result = image_patch.detect_chest_abnormality("chest_xray.png")
        >>> # Returns detection with bounding boxes + precise segmentation for each abnormality
        >>> num_abnormalities = len(result['detection']['boxes'])
        >>> for seg in result['segmentations']:
        ...     print(f"Found {{seg['abnormality']}} at {{seg['box']}}")
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
### Example 1: Chest X-Ray Disease Classification
Plan:
Step 1: Use classification_chest(image_path) to classify the chest X-ray - Returns disease label or None (establishes primary diagnosis)
Step 2: Use detect_chest_abnormality(image_path) to detect infiltration, consolidation, or lung opacity - Returns detection with bboxes and segmentation masks (provides anatomical evidence)
Step 3: Cross-validate results: check if classification indicates "pneumonia" AND detection found pneumonia-related abnormalities (infiltration/consolidation/lung opacity)
Step 4: If pneumonia indicators found, use verify_property with all images (original, detection overlays with bboxes, abnormality segmentation masks) to provide detailed clinical analysis: "Describe the pneumonia signs including anatomical location, extent of involvement, pattern (lobar vs diffuse), and severity assessment"
Step 5: Return comprehensive answer: "Yes/No pneumonia detected" with classification result, detected abnormalities with precise locations, segmentation visualizations, and detailed clinical assessment with recommendations

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Classify chest X-ray - returns list of diseases or None
    diagnoses = image_patch.classification_chest(image_path)

    # Step 2: Detect abnormalities (infiltration, consolidation, lung opacity)
    detection_result = image_patch.detect_chest_abnormality(image_path)

    # Step 3: Cross-validate - check if classification indicates pneumonia AND detection found related abnormalities
    has_pneumonia = False
    pneumonia_indicators = ['Infiltration', 'Consolidation', 'Lung Opacity']
    
    classification_has_pneumonia = diagnoses and any(indicator in diagnoses for indicator in pneumonia_indicators)
    detection_has_pneumonia = any(abnorm in pneumonia_indicators for abnorm in detection_result['detection']['label_names'])
    
    has_pneumonia = classification_has_pneumonia and detection_has_pneumonia

    # Step 4: Detailed clinical analysis if pneumonia indicators found
    detailed_analysis = None
    if has_pneumonia:
        # Collect all visual evidence: original image + detection overlays (with bboxes) + segmentation overlays (individual masks on image)
        overlay_images = [image_path] + list(detection_result['detection']['overlay_paths'].values())

        # Add individual segmentation overlays (each abnormality's mask overlaid on original image)
        overlay_images.extend([seg['overlay_path'] for seg in detection_result['segmentations']])

        detailed_analysis = image_patch.verify_property(
            overlay_images,
            f"Describe the pneumonia signs including anatomical location, extent of involvement, pattern (lobar vs diffuse), and severity assessment. Detected conditions: {{', '.join(diagnoses)}}. Focus on: {{', '.join(detection_result['detection']['label_names'])}}"
        )

    # Step 5: Return comprehensive answer
    return {{
        "pneumonia_detected": has_pneumonia,
        "all_diagnoses": diagnoses if diagnoses else [],
        "detected_abnormalities": detection_result['detection']['label_names'],
        "detection_overlays": detection_result['detection']['overlay_paths'],
        "segmentation_masks": [seg['mask_path'] for seg in detection_result['segmentations']],
        "detailed_analysis": detailed_analysis,
        "recommendation": "Urgent medical evaluation recommended" if has_pneumonia else "Continue monitoring"
    }}

### Example 2: Medical Knowledge Question
Plan:
Step 1: Return definition: "Pneumonia is an infection of the lung tissue causing inflammation of the air sacs (alveoli), which fill with fluid or pus. Common symptoms include cough, fever, difficulty breathing, and chest pain. It can be caused by bacteria, viruses, or fungi."

A:
def execute_command(image_path=None):
    return {{
        "question": "What is pneumonia?",
        "answer": "Pneumonia is an infection of the lung tissue causing inflammation of the air sacs (alveoli), which fill with fluid or pus. Common symptoms include cough, fever, difficulty breathing, and chest pain. It can be caused by bacteria, viruses, or fungi.",
        "type": "medical_knowledge"
    }}

### Example 3: Property Verification
Plan:
Step 1: Use classification_chest(image_path) to check if "cardiomegaly" is detected - Returns disease label or None (initial screening)
Step 2: Use detect_chest_abnormality(image_path) to detect cardiomegaly - Returns detection with precise bbox if present (confirms location and extent)
Step 3: Use segment_lungs_heart(image_path) to segment heart and lung fields - Returns H_mask_path, RL_mask_path, LL_mask_path (enables cardiothoracic ratio measurement)
Step 4: Use verify_property with all visual evidence (original image, cardiomegaly detection overlay if present, heart segmentation mask) to provide clinical assessment: "Evaluate the heart size and calculate the cardiothoracic ratio. Is the heart enlarged? Explain the findings with reference to normal values (CTR < 0.5)"
Step 5: Return definitive yes/no answer with: classification result, detection visualization with bbox, cardiothoracic ratio measurement from segmentation, and comprehensive clinical assessment with severity grading if enlarged

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Classify for cardiomegaly - returns list of diseases or None
    diagnoses = image_patch.classification_chest(image_path)

    # Step 2: Detect cardiomegaly with precise bbox
    detection_result = image_patch.detect_chest_abnormality(image_path)

    # Step 3: Segment heart and lung fields
    segmentation = image_patch.segment_lungs_heart(image_path)

    # Step 4: Clinical assessment with all visual evidence
    overlay_images = [image_path]

    # Add cardiomegaly detection overlay if present (bbox drawn on image)
    if detection_result['detection']['overlay_paths'] and 'Cardiomegaly' in detection_result['detection']['overlay_paths']:
        overlay_images.append(detection_result['detection']['overlay_paths']['Cardiomegaly'])

    # Add anatomical segmentation overlay (lungs/heart segmented on image)
    overlay_images.append(segmentation['overlay_path'])

    # Add individual abnormality segmentation overlays if any
    for seg in detection_result['segmentations']:
        if seg['abnormality'] == 'Cardiomegaly':
            overlay_images.append(seg['overlay_path'])

    assessment = image_patch.verify_property(
        overlay_images,
        f"Evaluate the heart size and calculate the cardiothoracic ratio. Is the heart enlarged? Explain the findings with reference to normal values (CTR < 0.5). Detected conditions: {{diagnoses if diagnoses else 'None'}}."
    )

    # Step 5: Return definitive yes/no answer
    is_enlarged = diagnoses and "Cardiomegaly" in diagnoses

    return {{
        "heart_enlarged": is_enlarged,
        "all_diagnoses": diagnoses if diagnoses else [],
        "detection_result": {{
            "has_cardiomegaly_bbox": "Cardiomegaly" in detection_result['detection']['label_names'],
            "overlay_path": detection_result['detection']['overlay_paths'].get('Cardiomegaly')
        }},
        "segmentation": {{
            "heart_mask": segmentation['H_mask_path'],
            "overlay": segmentation['overlay_path']
        }},
        "detailed_assessment": assessment,
        "recommendation": "Cardiology consultation recommended" if is_enlarged else "Heart size appears normal"
    }}

### Example 4: Detailed Chest Analysis
Plan:
Step 1: Use segment_lungs_heart(image_path) for anatomical segmentation - Returns lung and heart masks (establishes anatomical baseline)
Step 2: Use detect_chest_abnormality(image_path) to detect and segment any abnormalities - Returns detection with bboxes and segmentation masks for each abnormality (identifies all pathological findings)
Step 3: Use classification_chest(image_path) to identify primary disease/condition - Returns disease label or None (provides overall diagnostic impression)
Step 4: Use verify_property with comprehensive visual evidence (original image, anatomical segmentation overlay, all detection overlays) to generate detailed radiological report: "Provide comprehensive chest X-ray analysis including: 1) Technical quality, 2) Anatomical structures assessment, 3) All detected abnormalities with locations and characteristics, 4) Primary diagnosis, 5) Clinical significance and recommendations"
Step 5: Return structured comprehensive report with: anatomical segmentation visualizations, complete list of detected abnormalities with precise locations and segmentations, classification diagnosis, and detailed clinical narrative integrating all findings

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Anatomical segmentation
    segmentation = image_patch.segment_lungs_heart(image_path)

    # Step 2: Detect and segment any abnormalities
    detection_result = image_patch.detect_chest_abnormality(image_path)

    # Step 3: Classify diseases - returns list of diseases or None
    diagnoses = image_patch.classification_chest(image_path)

    # Step 4: Comprehensive analysis with all visual evidence
    overlay_images = [image_path, segmentation['overlay_path']]

    # Add all detection overlays (bboxes drawn on image)
    overlay_images.extend(list(detection_result['detection']['overlay_paths'].values()))

    # Add individual abnormality segmentation overlays (each abnormality's mask on image)
    overlay_images.extend([seg['overlay_path'] for seg in detection_result['segmentations']])

    findings_text = f"Detected conditions: {{', '.join(diagnoses)}}" if diagnoses else "No specific abnormalities detected"
    comprehensive_analysis = image_patch.verify_property(
        overlay_images,
        f"Provide comprehensive chest X-ray analysis including: 1) Technical quality, 2) Anatomical structures assessment, 3) All detected abnormalities with locations and characteristics, 4) Primary diagnosis, 5) Clinical significance and recommendations. {{findings_text}}. Analyze the segmentation overlays and detection results."
    )

    # Step 5: Return complete structured report
    return {{
        "all_findings": diagnoses if diagnoses else [],
        "num_findings": len(diagnoses) if diagnoses else 0,
        "anatomical_segmentation": {{
            "overlay": segmentation['overlay_path'],
            "right_lung": segmentation['RL_mask_path'],
            "left_lung": segmentation['LL_mask_path'],
            "heart": segmentation['H_mask_path']
        }},
        "detected_abnormalities": {{
            "count": len(detection_result['detection']['boxes']),
            "types": detection_result['detection']['label_names'],
            "detection_overlays": detection_result['detection']['overlay_paths'],
            "segmentation_masks": [
                {{
                    "abnormality": seg['abnormality'],
                    "mask_path": seg['mask_path'],
                    "overlay_path": seg['overlay_path'],
                    "location": seg['box']
                }}
                for seg in detection_result['segmentations']
            ]
        }},
        "comprehensive_analysis": comprehensive_analysis,
        "has_abnormality": diagnoses is not None and len(diagnoses) > 0,
        "recommendation": "Medical review recommended" if diagnoses else "No abnormalities detected - appears normal"
    }}
'''