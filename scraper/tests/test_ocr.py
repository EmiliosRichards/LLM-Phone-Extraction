import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image, UnidentifiedImageError, ImageError
import logging
import time
from scraper.ocr import ocr_image
from scraper.utils import download_and_process_image

# Configure logging for all tests
@pytest.fixture(autouse=True)
def setup_logging():
    logging.basicConfig(level=logging.DEBUG)

# Fixture for test image path
@pytest.fixture
def test_image_path():
    return Path("test_image.png")

# Fixture for mock image
@pytest.fixture
def mock_image():
    image = MagicMock(spec=Image.Image)
    image.format = "PNG"
    image.mode = "RGB"
    image.getbbox.return_value = (0, 0, 100, 100)  # Non-empty image
    image.size = (100, 100)
    image.convert.return_value = image
    return image

# Test successful OCR processing
def test_ocr_image_success(test_image_path, mock_image, caplog):
    with patch('PIL.Image.open', return_value=mock_image), \
         patch('pytesseract.image_to_string', return_value="Test OCR Result"):
        
        result = ocr_image(test_image_path)
        
        # Verify log messages
        assert "Starting OCR processing for" in caplog.records[0].message
        assert "Image format: PNG" in caplog.records[1].message
        assert "Image converted to grayscale" in caplog.records[2].message
        assert "Applied contrast enhancement and sharpening" in caplog.records[3].message
        assert "OCR completed: extracted 14 characters, 3 words" in caplog.records[4].message
        
        # Verify the result
        assert result["text"] == "Test OCR Result"
        assert result["char_count"] == 14
        assert result["word_count"] == 3
        assert result["path"] == str(test_image_path)

# Test empty image
def test_ocr_image_empty(test_image_path, mock_image, caplog):
    mock_image.getbbox.return_value = None
    
    with patch('PIL.Image.open', return_value=mock_image):
        with pytest.raises(ValueError):
            ocr_image(test_image_path)
        
        # Verify error log message
        assert "Image appears to be empty or corrupted" in caplog.records[0].message

# Test no text extracted
def test_ocr_image_no_text(test_image_path, mock_image, caplog):
    with patch('PIL.Image.open', return_value=mock_image), \
         patch('pytesseract.image_to_string', return_value=""):
        
        result = ocr_image(test_image_path)
        
        # Verify warning log message
        assert "No text extracted from" in caplog.records[0].message
        
        # Verify the result
        assert result["text"] == ""
        assert result["char_count"] == 0
        assert result["word_count"] == 0
        assert result["path"] == str(test_image_path)

# Test file not found
def test_ocr_image_file_not_found(test_image_path, caplog):
    with patch('PIL.Image.open', side_effect=FileNotFoundError("File not found")):
        result = ocr_image(test_image_path)
        
        # Verify error log message
        assert "Image file not found" in caplog.records[0].message
        
        # Verify the result
        assert result["text"] == ""
        assert result["char_count"] == 0
        assert result["word_count"] == 0
        assert result["path"] == str(test_image_path)

# Test fast processing with large image
def test_ocr_image_fast_processing_large(test_image_path, caplog):
    large_image = MagicMock(spec=Image.Image)
    large_image.format = "PNG"
    large_image.mode = "RGB"
    large_image.getbbox.return_value = (0, 0, 1200, 1200)
    large_image.size = (1200, 1200)
    large_image.convert.return_value = large_image
    
    with patch('PIL.Image.open', return_value=large_image), \
         patch('pytesseract.image_to_string', return_value="Test OCR Result"):
        
        result = ocr_image(test_image_path, fast_processing=True)
        
        # Verify log messages
        assert "Starting OCR processing for" in caplog.records[0].message
        assert "Skipping resize for large image" in caplog.records[1].message
        
        # Verify the result
        assert result["text"] == "Test OCR Result"
        assert result["char_count"] == 14
        assert result["word_count"] == 3
        
        # Verify that resize was not called
        assert not (hasattr(large_image, 'resize') and large_image.resize.called)

# Test fast processing with small image
def test_ocr_image_fast_processing_small(test_image_path, caplog):
    small_image = MagicMock(spec=Image.Image)
    small_image.format = "PNG"
    small_image.mode = "RGB"
    small_image.getbbox.return_value = (0, 0, 200, 200)
    small_image.size = (200, 200)
    small_image.convert.return_value = small_image
    
    with patch('PIL.Image.open', return_value=small_image), \
         patch('pytesseract.image_to_string', return_value="Test OCR Result"):
        
        result = ocr_image(test_image_path, fast_processing=True)
        
        # Verify log messages
        assert "Starting OCR processing for" in caplog.records[0].message
        assert "Image upscaled from" in caplog.records[1].message
        
        # Verify the result
        assert result["text"] == "Test OCR Result"
        assert result["char_count"] == 14
        assert result["word_count"] == 3
        
        # Verify that resize was called
        assert hasattr(small_image, 'resize') and small_image.resize.called

# Test non-image file
def test_ocr_image_non_image_file(test_image_path, caplog):
    with patch('PIL.Image.open', side_effect=UnidentifiedImageError("Cannot identify image file")):
        result = ocr_image(test_image_path)
        
        # Verify error log message
        assert "Unexpected error during OCR" in caplog.records[0].message
        assert "Cannot identify image file" in caplog.records[0].message
        
        # Verify the result
        assert result["text"] == ""
        assert result["char_count"] == 0
        assert result["word_count"] == 0
        assert result["path"] == str(test_image_path)

# Test Spanish text
def test_ocr_image_spanish(test_image_path, mock_image, caplog):
    spanish_text = "¡Hola! ¿Cómo estás? Bienvenido a mi casa."
    
    with patch('PIL.Image.open', return_value=mock_image), \
         patch('pytesseract.image_to_string', return_value=spanish_text):
        
        result = ocr_image(test_image_path)
        
        # Verify log messages
        assert "Starting OCR processing for" in caplog.records[0].message
        assert "OCR completed: extracted 45 characters, 6 words" in caplog.records[4].message
        
        # Verify the result
        assert result["text"] == spanish_text
        assert result["char_count"] == 45
        assert result["word_count"] == 6
        assert result["path"] == str(test_image_path)

# Test large text
def test_ocr_image_large_text(test_image_path, mock_image, caplog):
    large_text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.

Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt."""
    
    with patch('PIL.Image.open', return_value=mock_image), \
         patch('pytesseract.image_to_string', return_value=large_text):
        
        result = ocr_image(test_image_path)
        
        # Verify log messages
        assert "Starting OCR processing for" in caplog.records[0].message
        assert "OCR completed: extracted" in caplog.records[4].message
        
        # Verify the result
        assert result["text"] == large_text
        assert result["char_count"] == len(large_text)
        assert result["word_count"] == len(large_text.split())
        assert result["path"] == str(test_image_path)
        
        # Verify specific counts
        assert result["char_count"] == 1024  # Total characters including spaces and newlines
        assert result["word_count"] == 169   # Total words in the Lorem Ipsum text

# Test performance with large image
def test_ocr_image_performance(test_image_path, caplog):
    large_image = MagicMock(spec=Image.Image)
    large_image.format = "PNG"
    large_image.mode = "RGB"
    large_image.getbbox.return_value = (0, 0, 2000, 2000)  # Large 2000x2000 image
    large_image.size = (2000, 2000)
    large_image.convert.return_value = large_image
    
    with patch('PIL.Image.open', return_value=large_image), \
         patch('pytesseract.image_to_string', return_value="Test OCR Result"):
        
        # Set a reasonable timeout (5 seconds)
        timeout = 5.0
        
        # Measure execution time
        start_time = time.time()
        result = ocr_image(test_image_path, fast_processing=True)
        execution_time = time.time() - start_time
        
        # Verify performance
        assert execution_time < timeout, \
            f"OCR processing took {execution_time:.2f} seconds, exceeding timeout of {timeout} seconds"
        
        # Verify the result
        assert result["text"] == "Test OCR Result"
        assert result["char_count"] == 14
        assert result["word_count"] == 3
        assert result["path"] == str(test_image_path)
        
        # Log performance metrics
        logging.info(f"Large image OCR completed in {execution_time:.2f} seconds")

# Test OCR retry behavior
def test_ocr_image_with_retries(test_image_path, mock_image, caplog):
    with patch('time.sleep'), \
         patch('PIL.Image.open', return_value=mock_image), \
         patch('pytesseract.image_to_string', side_effect=[
             Exception("OCR failed attempt 1"),
             Exception("OCR failed attempt 2"),
             "Test OCR Result"  # Success on third try
         ]):
        
        result = download_and_process_image(
            full_url="http://example.com/test.png",
            img_path=test_image_path,
            ocr_retry_count=3,
            ocr_retry_delay=1.0
        )
        
        # Verify warning logs for retries
        assert "OCR attempt 1 failed" in caplog.records[0].message
        assert "Retrying OCR for" in caplog.records[1].message
        assert "OCR attempt 2 failed" in caplog.records[2].message
        assert "Retrying OCR for" in caplog.records[3].message
        
        # Verify the result
        assert result is not None
        assert result['ocr_text'] == "Test OCR Result"
        assert result['ocr_text_length'] == 14
        assert result['ocr_text_word_count'] == 3 