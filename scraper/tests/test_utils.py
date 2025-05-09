import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
from scraper.utils import validate_url, get_safe_filename, create_metadata

# Test data
VALID_URLS = [
    "https://example.com",
    "http://sub.example.com/path",
    "https://example.com/path?query=value",
    "http://example.com:8080/path",
]

INVALID_URLS = [
    ("", "URL must be a non-empty string"),
    ("not a url", "URL must include a scheme (e.g., 'http://' or 'https://')"),
    ("ftp://example.com", "Unsupported URL scheme: 'ftp'. Only 'http://' and 'https://' are supported"),
    ("http://", "URL must include a domain name"),
    ("http://example", "Invalid domain format: 'example'. Domain must contain at least one dot"),
    ("http://a.b", "Domain name too short: 'a.b'"),
    ("http://example.com/path with spaces", "URL path contains spaces. Please use URL encoding (e.g., %20)"),
    ("http://example.com/path//to/file", "URL path contains consecutive slashes"),
    ("http://example.com/path?query=value with spaces", "URL query contains spaces. Please use URL encoding (e.g., %20)"),
    ("http://example|.com", "Invalid domain format: 'example|'. Domain contains invalid characters"),
    ("http://example^.com", "Invalid domain format: 'example^'. Domain contains invalid characters"),
    ("http://example.com/path|with|pipes", "URL path contains invalid characters. Please use URL encoding"),
    ("http://example.com/path^with^carets", "URL path contains invalid characters. Please use URL encoding"),
]

# Test validate_url
@pytest.mark.parametrize("url", VALID_URLS)
def test_validate_url_valid(url):
    is_valid, error_msg = validate_url(url)
    assert is_valid
    assert error_msg == ""

@pytest.mark.parametrize("url,expected_error", INVALID_URLS)
def test_validate_url_invalid(url, expected_error):
    is_valid, error_msg = validate_url(url)
    assert not is_valid
    assert error_msg == expected_error

# Test get_safe_filename
def test_get_safe_filename_basic():
    url = "https://example.com/image.jpg"
    filename = get_safe_filename(url)
    assert filename.endswith(".jpg")
    assert "image_" in filename
    assert len(filename) > 8  # Should include hash

def test_get_safe_filename_no_extension():
    url = "https://example.com/image"
    filename = get_safe_filename(url)
    assert filename.endswith(".png")  # Default extension
    assert "image_" in filename

def test_get_safe_filename_invalid_chars():
    url = "https://example.com/path/with <invalid> chars/image.jpg"
    filename = get_safe_filename(url)
    assert "<" not in filename
    assert ">" not in filename
    assert " " not in filename
    assert filename.endswith(".jpg")

def test_get_safe_filename_long_url():
    long_path = "/" + "a" * 200 + ".jpg"
    url = f"https://example.com{long_path}"
    filename = get_safe_filename(url)
    assert len(filename) < 150  # Should be truncated
    assert filename.endswith(".jpg")

def test_get_safe_filename_empty_path():
    url = "https://example.com/"
    filename = get_safe_filename(url)
    assert filename.startswith("image_")
    assert filename.endswith(".png")

def test_get_safe_filename_encoded_chars():
    url = "https://example.com/path/image%20with%20spaces%26special.jpg"
    filename = get_safe_filename(url)
    assert "image_with_spaces_special" in filename
    assert filename.endswith(".jpg")
    assert "%" not in filename
    assert " " not in filename
    assert "&" not in filename

def test_get_safe_filename_unrecognized_extension():
    url = "https://example.com/image.xyz"
    filename = get_safe_filename(url)
    assert filename.endswith(".png")  # Should default to .png for unrecognized extensions
    assert "image_" in filename
    assert len(filename) > 8  # Should include hash

def test_get_safe_filename_no_path():
    url = "https://example.com"
    filename = get_safe_filename(url)
    assert filename.startswith("image_")
    assert filename.endswith(".png")
    assert len(filename) > 8  # Should include hash

# Test create_metadata
def test_create_metadata_basic():
    url = "https://example.com"
    hostname = "example.com"
    metadata = create_metadata(url, hostname)
    
    assert metadata["url"] == url
    assert metadata["hostname"] == hostname
    assert "timestamp" in metadata
    assert "text" not in metadata
    assert "ocr_results" not in metadata

def test_create_metadata_with_text():
    url = "https://example.com"
    hostname = "example.com"
    text = "Sample text content"
    metadata = create_metadata(url, hostname, text=text)
    
    assert metadata["text"] == text
    assert metadata["text_length"] == len(text)
    assert metadata["word_count"] == 3
    assert "ocr_results" not in metadata

@pytest.mark.parametrize("text,ocr_results,expected", [
    # Test case 1: Only OCR results
    (
        None,
        [
            {
                "image_url": "https://example.com/image1.jpg",
                "image_path": "image1.jpg",
                "ocr_text_length": 100,
                "ocr_text_word_count": 20
            },
            {
                "image_url": "https://example.com/image2.jpg",
                "image_path": "image2.jpg",
                "ocr_text_length": 200,
                "ocr_text_word_count": 40
            }
        ],
        {
            "total_images": 2,
            "total_ocr_text_length": 300,
            "total_ocr_word_count": 60,
            "ocr_results_length": 2,
            "has_text": False
        }
    ),
    # Test case 2: Both text and OCR results
    (
        "Sample text content",
        [
            {
                "image_url": "https://example.com/image1.jpg",
                "image_path": "image1.jpg",
                "ocr_text_length": 100,
                "ocr_text_word_count": 20
            }
        ],
        {
            "total_images": 1,
            "total_ocr_text_length": 100,
            "total_ocr_word_count": 20,
            "ocr_results_length": 1,
            "has_text": True,
            "text": "Sample text content",
            "text_length": 17,
            "word_count": 3
        }
    ),
    # Test case 3: Empty OCR results
    (
        None,
        [],
        {
            "total_images": 0,
            "total_ocr_text_length": 0,
            "total_ocr_word_count": 0,
            "ocr_results_length": 0,
            "has_text": False
        }
    ),
    # Test case 4: OCR result missing ocr_text_length
    (
        None,
        [
            {
                "image_url": "https://example.com/image1.jpg",
                "image_path": "image1.jpg",
                "ocr_text_word_count": 20
            }
        ],
        {
            "total_images": 1,
            "total_ocr_text_length": 0,  # Should default to 0
            "total_ocr_word_count": 20,
            "ocr_results_length": 1,
            "has_text": False
        }
    ),
    # Test case 5: OCR result missing ocr_text_word_count
    (
        None,
        [
            {
                "image_url": "https://example.com/image1.jpg",
                "image_path": "image1.jpg",
                "ocr_text_length": 100
            }
        ],
        {
            "total_images": 1,
            "total_ocr_text_length": 100,
            "total_ocr_word_count": 0,  # Should default to 0
            "ocr_results_length": 1,
            "has_text": False
        }
    ),
    # Test case 6: OCR result missing both text metrics
    (
        None,
        [
            {
                "image_url": "https://example.com/image1.jpg",
                "image_path": "image1.jpg"
            }
        ],
        {
            "total_images": 1,
            "total_ocr_text_length": 0,  # Should default to 0
            "total_ocr_word_count": 0,   # Should default to 0
            "ocr_results_length": 1,
            "has_text": False
        }
    )
])
def test_create_metadata_with_ocr_variations(text, ocr_results, expected, tmp_path):
    url = "https://example.com"
    hostname = "example.com"
    
    # Convert string paths to Path objects using tmp_path
    for result in ocr_results:
        result["image_path"] = tmp_path / result["image_path"]
    
    metadata = create_metadata(url, hostname, text=text, ocr_results=ocr_results)
    
    # Verify OCR-related fields
    assert metadata["total_images"] == expected["total_images"]
    assert metadata["total_ocr_text_length"] == expected["total_ocr_text_length"]
    assert metadata["total_ocr_word_count"] == expected["total_ocr_word_count"]
    assert len(metadata["ocr_results"]) == expected["ocr_results_length"]
    
    # Verify text-related fields if present
    if expected["has_text"]:
        assert metadata["text"] == expected["text"]
        assert metadata["text_length"] == expected["text_length"]
        assert metadata["word_count"] == expected["word_count"]
    else:
        assert "text" not in metadata

def test_create_metadata_timestamp_format():
    url = "https://example.com"
    hostname = "example.com"
    metadata = create_metadata(url, hostname)
    
    # Verify timestamp exists and is a string
    assert "timestamp" in metadata
    assert isinstance(metadata["timestamp"], str)
    
    # Verify timestamp is valid ISO 8601 format
    try:
        dt = datetime.fromisoformat(metadata["timestamp"])
        # Verify it's a recent timestamp (within last minute)
        now = datetime.now()
        time_diff = abs((now - dt).total_seconds())
        assert time_diff < 60, "Timestamp should be recent"
    except ValueError as e:
        pytest.fail(f"Timestamp is not valid ISO 8601 format: {e}") 