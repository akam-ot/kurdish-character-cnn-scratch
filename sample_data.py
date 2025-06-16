import os
import random
from typing import List, Tuple, Optional

# Sample data configuration
SAMPLES_DIR = "sample_images"

SAMPLE_DATA = {
    "01": ["01_sample.jpg"],
    "02": ["02_sample.jpg"],
    "03": ["03_sample.jpg"],
    "04": ["04_sample.jpg"],
    "05": ["05_sample.jpg"],
    "06": ["06_sample.jpg"],
    "07": ["07_sample.jpg"],
    "08": ["08_sample.jpg"],
    "09": ["09_sample.jpg"],
    "10": ["10_sample.jpg"],
    "11": ["11_sample.jpg"],
    "12": ["12_sample.jpg"],
    "13": ["13_sample.jpg"],
    "14": ["14_sample.jpg"],
    "15": ["15_sample.jpg"],
    "16": ["16_sample.jpg"],
    "17": ["17_sample.jpg"],
    "18": ["18_sample.jpg"],
    "19": ["19_sample.jpg"],
    "20": ["20_sample.jpg"],
    "21": ["21_sample.jpg"],
    "22": ["22_sample.jpg"],
    "23": ["23_sample.jpg"],
    "24": ["24_sample.jpg"],
    "25": ["25_sample.jpg"],
    "26": ["26_sample.jpg"],
    "27": ["27_sample.jpg"],
    "28": ["28_sample.jpg"],
    "29": ["29_sample.jpg"],
    "30": ["30_sample.jpg"],
    "31": ["31_sample.jpg"],
    "32": ["32_sample.jpg"],
    "33": ["33_sample.jpg"],
    "34": ["34_sample.jpg"],
    "35": ["35_sample.jpg"],
}


def get_available_samples() -> List[Tuple[str, str, str]]:
    """
    Get list of available sample images.
    Returns: List of (class_name, display_name, file_path) tuples
    """
    samples = []
    for class_name, files in SAMPLE_DATA.items():
        for file_name in files:
            file_path = os.path.join(SAMPLES_DIR, file_name)
            if os.path.exists(file_path):
                display_name = f"Class {class_name} - Sample {file_name.split('_')[-1].split('.')[0]}"
                samples.append((class_name, display_name, file_path))
    return samples


def get_samples_by_class(class_name: str) -> List[str]:
    """Get all sample file paths for a specific class."""
    if class_name not in SAMPLE_DATA:
        return []

    files = []
    for file_name in SAMPLE_DATA[class_name]:
        file_path = os.path.join(SAMPLES_DIR, file_name)
        if os.path.exists(file_path):
            files.append(file_path)
    return files


def get_random_sample() -> Optional[Tuple[str, str]]:
    """Get a random sample image."""
    available_samples = get_available_samples()
    if not available_samples:
        return None

    class_name, display_name, file_path = random.choice(available_samples)
    return display_name, file_path


def get_sample_info(file_path: str) -> Optional[str]:
    """Extract class information from sample file path."""
    file_name = os.path.basename(file_path)
    if "_" in file_name:
        class_name = file_name.split("_")[0]
        return class_name
    return None
