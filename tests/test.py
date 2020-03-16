import time
import numpy as np
from image_processing_library import ImageProcessingLibrary

def run(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    p = ImageProcessingLibrary()
    return p.run(image, kernel_size, sigma)

if __name__ == '__main__':
    source = np.load('source.npy')

    source_processed = np.load('source_processed.npy')

    start_time = time.time()
    target_processed = run(source, 15, 3).astype(np.uint8)
    print('Execution time: {0}'.format(time.time() - start_time))

    error = np.square(np.subtract(target_processed, source_processed)).mean()
    print('MSE: {0}'.format(error))
