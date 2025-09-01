from pipeline import CloudFactoryImageProcessor
import numpy as np
import os

def test_image_processor():
    processor = CloudFactoryImageProcessor()
    original, processed = processor.process_image("./images/Image1.jpg")
    assert isinstance(original, np.ndarray)
    assert isinstance(processed, np.ndarray)
    assert processed.shape[0] > 0 