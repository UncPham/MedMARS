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
            dict: {
                'image1.jpg': {
                    'label1': 99.99,  # confidence score (0-100)
                    'label2': 0.01,
                    'label3': 0.0,
                    ...
                },
                'image2.jpg': {...},
                ...
            }

        Example:
            Input: images=['img1.jpg'], labels=['chest X-ray', 'brain MRI']
            Output: {'img1.jpg': {'chest X-ray': 99.5, 'brain MRI': 0.5}}

    segment_lungs_heart(image_path: str) → dict
        Segments lungs and heart in chest X-ray images.

        Returns:
            dict: {
                'overlay_path': str,   # Path to colored overlay visualization
                'RL_mask_path': str,   # Right lung segmentation mask
                'LL_mask_path': str,   # Left lung segmentation mask
                'H_mask_path': str     # Heart segmentation mask
            }

        Example: {
            'overlay_path': '/path/to/chest_overlay.png',
            'RL_mask_path': '/path/to/right_lung.png',
            'LL_mask_path': '/path/to/left_lung.png',
            'H_mask_path': '/path/to/heart.png'
        }

    segment_bowel_stomach(image_path: str) → dict
        Segments bowel and stomach regions in abdominal images.

        Returns:
            dict: {
                'overlay_path': str,              # Overlay visualization
                'large_bowel_mask_path': str,     # Large bowel mask
                'small_bowel_mask_path': str,     # Small bowel mask
                'stomach_mask_path': str,         # Stomach mask
                'segmentation_info': [            # Statistics for each organ
                    {
                        'class': str,             # e.g., 'large bowel'
                        'percentage': float,      # % of image covered
                        'color': tuple           # RGB color (R, G, B)
                    },
                    ...
                ]
            }

        Example: {
            'overlay_path': '/path/to/abdomen_overlay.png',
            'large_bowel_mask_path': '/path/to/large_bowel.png',
            'small_bowel_mask_path': '/path/to/small_bowel.png',
            'stomach_mask_path': '/path/to/stomach.png',
            'segmentation_info': [
                {'class': 'large bowel', 'percentage': 15.3, 'color': (0, 0, 255)},
                {'class': 'small bowel', 'percentage': 22.1, 'color': (23, 154, 0)},
                {'class': 'stomach', 'percentage': 8.7, 'color': (255, 127, 0)}
            ]
        }

    detect_brain_tumor(image_path: str) → dict
        Detects and segments brain tumors in MRI/CT scans using YOLO + MedSAM.
        Can detect 3 tumor types: glioma, meningioma, pituitary.

        Returns:
            dict: {
                'detection_path': str,        # Visualization with bounding boxes
                'num_detections': int,        # Total number of tumors found
                'detections': [               # List of detected tumors
                    {
                        'class': str,         # 'glioma'/'meningioma'/'pituitary'
                        'confidence': float,  # Detection confidence (0.0-1.0)
                        'bbox': [x1, y1, x2, y2]  # Bounding box coordinates
                    },
                    ...
                ],
                'segmentations': [            # Segmentation for each detection
                    {
                        'detection_index': int,      # Index in detections list
                        'bbox': [x1, y1, x2, y2],   # Same bbox as detection
                        'mask_path': str,            # Binary segmentation mask
                        'overlay_path': str          # Mask overlaid on image
                    },
                    ...
                ]
            }

        Example: {
            'detection_path': '/path/to/brain_detection.png',
            'num_detections': 2,
            'detections': [
                {'class': 'glioma', 'confidence': 0.92, 'bbox': [120, 80, 180, 140]},
                {'class': 'meningioma', 'confidence': 0.87, 'bbox': [200, 150, 250, 200]}
            ],
            'segmentations': [
                {
                    'detection_index': 0,
                    'bbox': [120, 80, 180, 140],
                    'mask_path': '/path/to/tumor_0_mask.png',
                    'overlay_path': '/path/to/tumor_0_overlay.png'
                },
                {
                    'detection_index': 1,
                    'bbox': [200, 150, 250, 200],
                    'mask_path': '/path/to/tumor_1_mask.png',
                    'overlay_path': '/path/to/tumor_1_overlay.png'
                }
            ]
        }

    verify_property(image_path: str, query: str) → str
        Verifies specific visual properties or answers detailed questions about the image.
        Returns natural language explanation/answer.

        Use for:
        - Property verification: "is the lesion raised?", "does it have irregular borders?"
        - Descriptive questions: "what is the texture?", "what color is the affected area?"
        - Complex assessments: "are there signs of inflammation?"
        - Comparative queries: "is this larger than normal?"
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
        >>> # Returns: {
        >>> #   "img1.png": {"pneumonia": 92.5, "normal": 5.2, "chest X-ray": 2.3},
        >>> #   "img2.png": {"pneumonia": 3.1, "normal": 95.8, "chest X-ray": 1.1}
        >>> # }
        >>> # Each image gets confidence scores (0-100) for ALL provided labels
        """
        pass

    def segment_lungs_heart(self, image_path: str) -> dict:
        """
        Example:
        >>> result = image_patch.segment_lungs_heart("chest_xray.png")
        >>> # Returns: {
        >>> #   'overlay_path': '/path/to/overlay.png',
        >>> #   'RL_mask_path': '/path/to/right_lung.png',
        >>> #   'LL_mask_path': '/path/to/left_lung.png',
        >>> #   'H_mask_path': '/path/to/heart.png'
        >>> # }
        """
        pass

    def segment_bowel_stomach(self, image_path: str) -> dict:
        """
        Example:
        >>> result = image_patch.segment_bowel_stomach("abdomen_scan.png")
        >>> # Returns: {
        >>> #   'overlay_path': '/path/to/abdomen_overlay.png',
        >>> #   'large_bowel_mask_path': '/path/to/large_bowel.png',
        >>> #   'small_bowel_mask_path': '/path/to/small_bowel.png',
        >>> #   'stomach_mask_path': '/path/to/stomach.png',
        >>> #   'segmentation_info': [
        >>> #       {'class': 'large bowel', 'percentage': 15.3, 'color': (0, 0, 255)},
        >>> #       {'class': 'small bowel', 'percentage': 22.1, 'color': (23, 154, 0)},
        >>> #       {'class': 'stomach', 'percentage': 8.7, 'color': (255, 127, 0)}
        >>> #   ]
        >>> # }
        """
        pass

    def detect_brain_tumor(self, image_path: str) -> dict:
        """
        Example:
        >>> result = image_patch.detect_brain_tumor("brain_mri.png")
        >>> # Returns: {
        >>> #   'num_detections': 2,
        >>> #   'detection_path': '/path/to/detection.png',
        >>> #   'detections': [{bbox: [x1,y1,x2,y2]}, ...],
        >>> #   'segmentations': [{mask_path: '...', overlay_path: '...'}, ...]
        >>> # }
        """
        pass

    def verify_property(self, image_path: str, query: str) -> str:
        """
        Example:
        >>> answer = image_patch.verify_property(
        ...     image_path,
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
Step 1: Use classification_chest(image_path) to classify the chest X-ray
Step 2: If result is "pneumonia", use verify_property to get detailed analysis
Step 3: Use segment_lungs_heart to visualize affected regions
Step 4: Return comprehensive results

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Classify chest X-ray
    diagnosis = image_patch.classification_chest(image_path)

    # Step 2: Detailed analysis if pneumonia detected
    detailed_analysis = None
    if diagnosis == "pneumonia":
        detailed_analysis = image_patch.verify_property(
            image_path,
            "Describe the pneumonia signs in this chest X-ray including location, extent, and pattern"
        )

    # Step 3: Segment lungs and heart for visualization
    segmentation = image_patch.segment_lungs_heart(image_path)

    # Step 4: Return results
    return {
        "pneumonia_detected": diagnosis == "pneumonia",
        "diagnosis": diagnosis,
        "detailed_analysis": detailed_analysis,
        "segmentation_paths": segmentation,
    }

### Example 2: Brain Tumor Detection and Analysis
Plan:
Step 1: Use detect_brain_tumor(image_path) to detect and segment tumors
Step 2: If tumors detected, use verify_property for detailed characterization
Step 3: Return tumor count, visualizations, and detailed description

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Detect and segment tumors
    tumor_result = image_patch.detect_brain_tumor(image_path)

    # Step 2: Detailed analysis if tumors detected
    detailed_description = None
    if tumor_result['num_detections'] > 0:
        detailed_description = image_patch.verify_property(
            image_path,
            "Describe the characteristics of the detected brain tumors including size, location, and appearance"
        )

    # Step 3: Return comprehensive results
    return {
        "num_tumors": tumor_result['num_detections'],
        "tumors_detected": tumor_result['num_detections'] > 0,
        "detection_visualization": tumor_result['detection_path'],
        "segmentations": tumor_result['segmentations'],
        "detailed_description": detailed_description,
    }

### Example 3: Organ Classification
Plan:
Step 1: Use classification_organa(image_path) to classify the organ
Step 2: If classification successful, use verify_property for detailed description
Step 3: Return organ name and description

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Classify organ
    organ = image_patch.classification_organa(image_path)

    # Step 2: Get detailed description if classification succeeded
    description = None
    if organ:
        description = image_patch.verify_property(
            image_path,
            "Describe the characteristics and appearance of this organ"
        )

    # Step 3: Return results
    return {
        "organ_identified": organ is not None,
        "organ": organ if organ else "Unknown",
        "description": description,
        "recommendation": "Organ identified successfully" if organ else "Unable to identify organ with confidence"
    }

### Example 6: Medical Knowledge Question
Plan:
Step 1: Return definition of pneumonia

A:
def execute_command(image_path=None):
    return {
        "question": "What is pneumonia?",
        "answer": "Pneumonia is an infection of the lung tissue causing inflammation of the air sacs (alveoli), which fill with fluid or pus. Common symptoms include cough, fever, difficulty breathing, and chest pain. It can be caused by bacteria, viruses, or fungi.",
        "type": "medical_knowledge"
    }

### Example 7: Heart Enlargement Assessment
Plan:
Step 1: Use classification_chest to check for cardiomegaly
Step 2: Use segment_lungs_heart to visualize heart
Step 3: Use verify_property for detailed assessment
Step 4: Return yes/no answer with supporting evidence

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Classify for cardiomegaly
    diagnosis = image_patch.classification_chest(image_path)

    # Step 2: Segment heart
    segmentation = image_patch.segment_lungs_heart(image_path)

    # Step 3: Detailed assessment
    assessment = image_patch.verify_property(
        image_path,
        "Assess the heart size and calculate the cardiothoracic ratio. Is the heart enlarged?"
    )

    # Step 4: Return comprehensive answer
    is_enlarged = diagnosis == "cardiomegaly"

    return {
        "heart_enlarged": is_enlarged,
        "classification": diagnosis,
        "heart_mask": segmentation['H_mask_path'],
        "overlay": segmentation['overlay_path'],
        "detailed_assessment": assessment,
        "recommendation": "Cardiology consultation recommended" if is_enlarged else "Heart size appears normal"
    }

### Example 8: Comprehensive Chest X-Ray Analysis
Plan:
Step 1: Use classification_chest to identify primary findings
Step 2: Use segment_lungs_heart for anatomical segmentation
Step 3: Use verify_property for comprehensive analysis
Step 4: Return complete report

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Classify disease
    disease = image_patch.classification_chest(image_path)

    # Step 2: Anatomical segmentation
    segmentation = image_patch.segment_lungs_heart(image_path)

    # Step 3: Comprehensive analysis
    comprehensive_analysis = image_patch.verify_property(
        image_path,
        "Provide a comprehensive analysis of this chest X-ray including: anatomical structures visible, any abnormalities, their location and characteristics, and clinical significance"
    )

    # Step 4: Return complete report
    return {
        "primary_finding": disease if disease else "No specific disease classified",
        "anatomical_segmentation": {
            "overlay": segmentation['overlay_path'],
            "right_lung": segmentation['RL_mask_path'],
            "left_lung": segmentation['LL_mask_path'],
            "heart": segmentation['H_mask_path']
        },
        "comprehensive_analysis": comprehensive_analysis,
        "has_abnormality": disease is not None,
        "recommendation": "Medical review recommended" if disease else "Appears normal"
    }

### Example 9: Tumor Count and Localization
Plan:
Step 1: Use detect_brain_tumor to detect all tumors
Step 2: Use verify_property for detailed location descriptions
Step 3: Return count, locations, and visualizations

A:
def execute_command(image_path):
    image_patch = ImagePatch()

    # Step 1: Detect tumors
    tumor_result = image_patch.detect_brain_tumor(image_path)

    # Step 2: Get location descriptions if tumors found
    location_description = None
    if tumor_result['num_detections'] > 0:
        location_description = image_patch.verify_property(
            image_path,
            "Describe the location of each detected tumor in anatomical terms (e.g., frontal lobe, parietal region, etc.)"
        )

    # Step 3: Return comprehensive results
    return {
        "tumor_count": tumor_result['num_detections'],
        "detection_visualization": tumor_result['detection_path'],
        "tumor_bounding_boxes": tumor_result['detections'],
        "segmentation_masks": tumor_result['segmentations'],
        "anatomical_locations": location_description,
        "urgency": "HIGH" if tumor_result['num_detections'] > 0 else "NONE"
    }

### Example 10: Comparative Disease Severity
Plan:
Step 1: Classify both X-rays
Step 2: Segment both for visualization
Step 3: Use verify_property to compare severity
Step 4: Return comparative analysis

A:
def execute_command(image_paths):
    image_patch = ImagePatch()
    image1_path, image2_path = image_paths[0], image_paths[1]

    # Step 1: Classify both images
    disease1 = image_patch.classification_chest(image1_path)
    disease2 = image_patch.classification_chest(image2_path)

    # Step 2: Segment both
    seg1 = image_patch.segment_lungs_heart(image1_path)
    seg2 = image_patch.segment_lungs_heart(image2_path)

    # Step 3: Comparative analysis
    comparison = image_patch.verify_property(
        [image1_path, image2_path],
        "Compare these two chest X-rays and assess which shows more severe disease based on extent of abnormalities, distribution, and radiological signs"
    )

    # Step 4: Return comparative results
    return {
        "image1": {
            "disease": disease1,
            "segmentation": seg1
        },
        "image2": {
            "disease": disease2,
            "segmentation": seg2
        },
        "comparative_analysis": comparison,
        "both_abnormal": disease1 is not None and disease2 is not None
    }
'''