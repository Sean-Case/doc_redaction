import pytest
from unittest.mock import patch, MagicMock
from PIL import Image, ImageDraw
import io
import os

# Assuming functions are in a module named 'file_redaction' in the parent directory
# Adjust the import path as necessary based on your project structure
from file_redaction import redact_image_pdf, redact_text_pdf, choose_and_run_redactor # Added choose_and_run_redactor
from presidio_analyzer import RecognizerResult # For mocking analyzer results

# Helper function to create a dummy PDF file (in-memory) with images
def create_dummy_image_pdf(num_pages=1, pii_locations=None):
    """
    Creates an in-memory PDF with a specified number of pages.
    Each page will be a simple image with optional black boxes to simulate PII.
    pii_locations: A list of lists, where each inner list contains (x, y, width, height) tuples for PII boxes on that page.
                   Example: [[(10,10,50,20)], [(20,30,40,10), (60,70,30,15)]] for a 2-page PDF.
    """
    pdf_images = []
    for i in range(num_pages):
        img = Image.new('RGB', (600, 800), color = 'white')
        draw = ImageDraw.Draw(img)
        draw.text((10, (i+1) * 20), f"Image Page {i+1}", fill='black') # Add page number text

        if pii_locations and i < len(pii_locations):
            for pii_box in pii_locations[i]:
                x, y, w, h = pii_box
                draw.rectangle([x, y, x + w, y + h], fill='black') # Simulate PII as black boxes
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG') # Save as PNG to be added to PDF
        img_byte_arr.seek(0)
        pdf_images.append(Image.open(img_byte_arr))

    if not pdf_images:
        # Create a blank page if nothing else to ensure valid PDF
        img = Image.new('RGB', (600, 800), color = 'white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        pdf_images.append(Image.open(img_byte_arr))


    # Save images to a PDF (in-memory)
    pdf_bytes_io = io.BytesIO()
    pdf_images[0].save(pdf_bytes_io, format="PDF", save_all=True, append_images=pdf_images[1:])
    pdf_bytes_io.seek(0)
    return pdf_bytes_io

# Helper function to create a dummy PDF with text
import fitz # PyMuPDF
def create_dummy_text_pdf(num_pages=1, page_texts=None):
    """
    Creates an in-memory PDF with a specified number of pages and text.
    page_texts: A list of strings, where each string is the text for that page.
                Example: ["My name is John Doe.", "My email is test@example.com"]
    """
    doc = fitz.open() # New empty PDF
    for i in range(num_pages):
        page = doc.new_page() # Add a new page
        text_to_insert = f"Text Page {i+1}"
        if page_texts and i < len(page_texts):
            text_to_insert = page_texts[i]
        # Insert text. Default font is Courier. Adjust rect as needed.
        page.insert_text(fitz.Point(50, 72), text_to_insert, fontsize=11)
    
    pdf_bytes_io = io.BytesIO()
    doc.save(pdf_bytes_io)
    doc.close()
    pdf_bytes_io.seek(0)
    return pdf_bytes_io


# --- Mock classes ---
class MockPIIEntity: # For Image PII
    def __init__(self, score, entity_type, left, top, width, height):
        self.score = score
        self.entity_type = entity_type
        self.left = left
        self.top = top
        self.width = width
        self.height = height

class MockImageAnalyzerEngine: # For Image PII
    def __init__(self, supported_entities=None):
        self.supported_entities = supported_entities or ["CREDIT_CARD", "SSN"]

    def analyze_image(self, image_bytes, entities):
        results = []
        if "SSN" in entities and "SSN" in self.supported_entities :
            results.append(MockPIIEntity(0.95, "SSN", 0.1, 0.1, 0.2, 0.05))
        if "CREDIT_CARD" in entities and "CREDIT_CARD" in self.supported_entities:
            results.append(MockPIIEntity(0.80, "CREDIT_CARD", 0.3, 0.3, 0.4, 0.08))
        return results

# Mock for Presidio AnalyzerEngine
class MockTextAnalyzerEngine:
    def __init__(self):
        pass # Not much setup needed as analyze will be a MagicMock

    def analyze(self, text, language, entities, score_threshold):
        # This method will be replaced by a MagicMock in the fixture.
        # The actual mock RecognizerResult objects will be configured in individual tests.
        return []


@pytest.fixture
def mock_s3_client():
    with patch('boto3.client') as mock_boto_client:
        s3_mock = MagicMock()
        
        def mock_download_file(Bucket, Key, Filename):
            dummy_pdf_bytes = None
            if "image_pdf_single_page_pii" in Key:
                pii_locs = [[(10, 10, 50, 20)]]
                dummy_pdf_bytes = create_dummy_image_pdf(num_pages=1, pii_locations=pii_locs)
            elif "image_pdf_multi_page_pii" in Key:
                pii_locs = [[(10,10,50,20)], [], [(20,30,40,10)]]
                dummy_pdf_bytes = create_dummy_image_pdf(num_pages=3, pii_locations=pii_locs)
            elif "image_pdf_multi_page_no_pii" in Key:
                dummy_pdf_bytes = create_dummy_image_pdf(num_pages=2)
            elif "image_pdf_single_page_no_pii" in Key:
                 dummy_pdf_bytes = create_dummy_image_pdf(num_pages=1)
            # Text PDF cases
            elif "text_pdf_single_page_pii" in Key:
                page_texts = ["My name is Alice and my number is 012-345-6789."]
                dummy_pdf_bytes = create_dummy_text_pdf(num_pages=1, page_texts=page_texts)
            elif "text_pdf_multi_page_pii" in Key:
                page_texts = ["Page 1: My name is Bob Smith.", "Page 2: My email is bob.smith@example.com and phone is 1234567890"]
                dummy_pdf_bytes = create_dummy_text_pdf(num_pages=2, page_texts=page_texts)
            elif "text_pdf_single_page_no_pii" in Key:
                page_texts = ["This page has no PII."]
                dummy_pdf_bytes = create_dummy_text_pdf(num_pages=1, page_texts=page_texts)
            elif "text_pdf_multi_page_no_pii" in Key:
                page_texts = ["Hello world.", "This is a test."]
                dummy_pdf_bytes = create_dummy_text_pdf(num_pages=2, page_texts=page_texts)
            else: # Default fallback: simple image pdf
                dummy_pdf_bytes = create_dummy_image_pdf(num_pages=1)

            with open(Filename, 'wb') as f:
                f.write(dummy_pdf_bytes.getvalue())

        s3_mock.download_file.side_effect = mock_download_file
        s3_mock.upload_fileobj = MagicMock()
        
        mock_boto_client.return_value = s3_mock
        yield s3_mock

@pytest.fixture
def mock_image_analyzer(): # For redact_image_pdf tests
    test_analyzer_instance = MockImageAnalyzerEngine()
    test_analyzer_instance.analyze_image = MagicMock() 
    yield test_analyzer_instance

@pytest.fixture
def mock_text_analyzer(): # For redact_text_pdf tests
    # This fixture provides a mock Presidio AnalyzerEngine.
    # The actual AnalyzerEngine is instantiated within redact_text_pdf, so we patch it there.
    with patch('file_redaction.AnalyzerEngine') as MockPresidioAnalyzer:
        mock_analyzer_instance = MockPresidioAnalyzer.return_value
        # The analyze method of the instance is what we want to control
        mock_analyzer_instance.analyze = MagicMock()
        yield mock_analyzer_instance # This is the mock of the *instance* of AnalyzerEngine

@pytest.fixture
def mock_anonymizer_engine(): # For redact_text_pdf tests
    with patch('file_redaction.AnonymizerEngine') as MockAnonymizer:
        mock_anonymizer_instance = MockAnonymizer.return_value
        # Mock the anonymize method
        # It should return an object with a 'text' attribute
        mock_anonymizer_instance.anonymize = MagicMock(
            return_value=MagicMock(text="[ANONYMIZED_TEXT]") 
        )
        yield mock_anonymizer_instance


# --- Test Cases for redact_image_pdf ---

def test_redact_single_page_pdf_with_pii(mock_s3_client, mock_image_analyzer):
    """Test redacting a single-page IMAGE PDF with PII."""
    input_bucket = "test-bucket"
    input_key = "image_pdf_single_page_pii.pdf" # Updated key
    output_bucket = "output-bucket"
    output_key = "redacted_image_pdf_single_page_pii.pdf"
    
    # Define the expected PII for the mock analyzer to return for this test
    # These coordinates are relative to page dimensions (0.0 to 1.0)
    mock_pii_for_page1 = [MockPIIEntity(0.9, "SSN", 0.016, 0.0125, 0.083, 0.025)] 
    mock_image_analyzer.analyze_image.return_value = mock_pii_for_page1

    result_key = redact_image_pdf( # This is testing redact_image_pdf
        input_s3_bucket_name=input_bucket,
        input_s3_key_name=input_key,
        output_s3_bucket_name=output_bucket,
        output_s3_key_name=output_key,
        image_analyzer_engine=mock_image_analyzer, # This is the mock from the fixture
        s3_client=mock_s3_client,
        min_score_threshold=0.85
    )

    assert result_key == output_key
    mock_s3_client.download_file.assert_called_once()
    args, kwargs = mock_s3_client.download_file.call_args
    assert args[0] == input_bucket
    assert args[1] == input_key
    assert isinstance(args[2], str) 

    mock_image_analyzer.analyze_image.assert_called() 
    mock_s3_client.upload_fileobj.assert_called_once()
    # Check that upload_fileobj was called for the output bucket and key
    # upload_fileobj is called as: s3_client.upload_fileobj(f, output_s3_bucket_name, output_s3_key_name)
    upload_call_args = mock_s3_client.upload_fileobj.call_args
    assert upload_call_args is not None, "upload_fileobj was not called"
    assert len(upload_call_args.args) >= 3, "upload_fileobj called with insufficient positional arguments"
    assert upload_call_args.args[1] == output_bucket # Bucket is the second positional argument
    assert upload_call_args.args[2] == output_key   # Key is the third positional argument

    # Further checks:
    # - Verify the content of the uploaded file (e.g., PII areas are redacted)
    #   This would require capturing the BytesIO object passed to upload_fileobj
    #   and using a PDF library to inspect it. For simplicity, we're focusing on interactions here.
    # - Check that the mock_image_analyzer was called with the correct image data (more complex).

def test_redact_multi_page_pdf_with_pii(mock_s3_client, mock_image_analyzer):
    """Test redacting a multi-page IMAGE PDF with PII on different pages."""
    input_bucket = "test-bucket"
    input_key = "image_pdf_multi_page_pii.pdf" # Updated key
    output_bucket = "output-bucket"
    output_key = "redacted_image_pdf_multi_page_pii.pdf"

    # Simulate different PII findings for different pages
    # Page 1 has PII, Page 2 no PII, Page 3 has PII
    mock_pii_page1 = [MockPIIEntity(0.95, "SSN", 0.016, 0.0125, 0.083, 0.025)] 
    mock_pii_page3 = [MockPIIEntity(0.90, "CREDIT_CARD", 0.033, 0.0375, 0.066, 0.0125)] 
    
    mock_image_analyzer.analyze_image.side_effect = [
        mock_pii_page1, 
        [],             
        mock_pii_page3  
    ]

    result_key = redact_image_pdf( # This is testing redact_image_pdf
        input_s3_bucket_name=input_bucket,
        input_s3_key_name=input_key,
        output_s3_bucket_name=output_bucket,
        output_s3_key_name=output_key,
        image_analyzer_engine=mock_image_analyzer, # This is the mock from the fixture
        s3_client=mock_s3_client,
        min_score_threshold=0.8
    )

    assert result_key == output_key
    mock_s3_client.download_file.assert_called_once()
    assert mock_image_analyzer.analyze_image.call_count == 3 # Called for each page
    mock_s3_client.upload_fileobj.assert_called_once()
    upload_call_args = mock_s3_client.upload_fileobj.call_args
    assert upload_call_args is not None, "upload_fileobj was not called"
    assert len(upload_call_args.args) >= 3, "upload_fileobj called with insufficient positional arguments"
    assert upload_call_args.args[1] == output_bucket
    assert upload_call_args.args[2] == output_key


def test_redact_pdf_with_no_pii(mock_s3_client, mock_image_analyzer):
    """Test redacting an IMAGE PDF that contains no PII."""
    input_bucket = "test-bucket"
    input_key = "image_pdf_single_page_no_pii.pdf" # Updated key
    output_bucket = "output-bucket"
    output_key = "redacted_image_pdf_single_page_no_pii.pdf"

    mock_image_analyzer.analyze_image.return_value = [] 

    result_key = redact_image_pdf( # This is testing redact_image_pdf
        input_s3_bucket_name=input_bucket,
        input_s3_key_name=input_key,
        output_s3_bucket_name=output_bucket,
        output_s3_key_name=output_key,
        image_analyzer_engine=mock_image_analyzer, # This is the mock from the fixture
        s3_client=mock_s3_client,
        min_score_threshold=0.9
    )

    assert result_key == output_key
    mock_s3_client.download_file.assert_called_once()
    mock_image_analyzer.analyze_image.assert_called_once() # Called for the single page
    mock_s3_client.upload_fileobj.assert_called_once()
    upload_call_args = mock_s3_client.upload_fileobj.call_args
    assert upload_call_args is not None, "upload_fileobj was not called"
    assert len(upload_call_args.args) >= 3, "upload_fileobj called with insufficient positional arguments"
    assert upload_call_args.args[1] == output_bucket
    assert upload_call_args.args[2] == output_key
    # In this case, the uploaded PDF should be identical to the input if no PII was found.
    # Capturing the uploaded file and comparing it would be a good addition.

def test_redact_multi_page_pdf_no_pii(mock_s3_client, mock_image_analyzer):
    """Test redacting a multi-page IMAGE PDF with no PII on any page."""
    input_bucket = "test-bucket"
    input_key = "image_pdf_multi_page_no_pii.pdf" # Updated key
    output_bucket = "output-bucket"
    output_key = "redacted_image_pdf_multi_page_no_pii.pdf"

    mock_image_analyzer.analyze_image.return_value = [] 

    result_key = redact_image_pdf( # This is testing redact_image_pdf
        input_s3_bucket_name=input_bucket,
        input_s3_key_name=input_key,
        output_s3_bucket_name=output_bucket,
        output_s3_key_name=output_key,
        image_analyzer_engine=mock_image_analyzer, # This is the mock from the fixture
        s3_client=mock_s3_client,
        min_score_threshold=0.9
    )

    assert result_key == output_key
    mock_s3_client.download_file.assert_called_once()
    assert mock_image_analyzer.analyze_image.call_count == 2 
    mock_s3_client.upload_fileobj.assert_called_once()
    upload_call_args = mock_s3_client.upload_fileobj.call_args
    assert upload_call_args is not None, "upload_fileobj was not called"
    assert len(upload_call_args.args) >= 3, "upload_fileobj called with insufficient positional arguments"
    assert upload_call_args.args[1] == output_bucket
    assert upload_call_args.args[2] == output_key

# --- Test Cases for redact_text_pdf ---

def test_redact_single_page_text_pdf_with_pii(mock_s3_client, mock_text_analyzer, mock_anonymizer_engine):
    """Test redacting a single-page TEXT PDF with PII."""
    input_bucket = "test-text-bucket"
    input_key = "text_pdf_single_page_pii.pdf"
    output_bucket = "output-text-bucket"
    output_key = "redacted_text_pdf_single_page_pii.pdf"
    
    # Mock PII detection for the text content
    # create_dummy_text_pdf for "text_pdf_single_page_pii.pdf" creates:
    # "My name is Alice and my number is 012-345-6789."
    # Analyzer should find "Alice" (PERSON) and "012-345-6789" (PHONE_NUMBER)
    mock_analyzer_results = [
        RecognizerResult(entity_type="PERSON", start=11, end=16, score=0.95), # "Alice"
        RecognizerResult(entity_type="PHONE_NUMBER", start=30, end=42, score=0.85) # "012-345-6789"
    ]
    mock_text_analyzer.analyze.return_value = mock_analyzer_results

    # Mock anonymizer behavior (optional, default mock anonymizes to "[ANONYMIZED_TEXT]")
    # For more specific testing, one could configure mock_anonymizer_engine.anonymize.return_value.text

    result_key = redact_text_pdf(
        input_s3_bucket_name=input_bucket,
        input_s3_key_name=input_key,
        output_s3_bucket_name=output_bucket,
        output_s3_key_name=output_key,
        s3_client=mock_s3_client,
        analyzer=mock_text_analyzer, # This is the mock_analyzer_instance from the fixture
        anonymizer=mock_anonymizer_engine,
        min_score_threshold=0.8,
        pii_entities=["PERSON", "PHONE_NUMBER"]
    )

    assert result_key == output_key
    mock_s3_client.download_file.assert_called_once()
    args, _ = mock_s3_client.download_file.call_args
    assert args[0] == input_bucket
    assert args[1] == input_key

    mock_text_analyzer.analyze.assert_called_once()
    analyze_args, analyze_kwargs = mock_text_analyzer.analyze.call_args
    # PyMuPDF's get_text("text") often adds a trailing newline
    assert analyze_kwargs['text'] == "My name is Alice and my number is 012-345-6789.\n"
    assert analyze_kwargs['entities'] == ["PERSON", "PHONE_NUMBER"]
    
    mock_anonymizer_engine.anonymize.assert_called_once()

    mock_s3_client.upload_fileobj.assert_called_once()
    upload_call_args = mock_s3_client.upload_fileobj.call_args
    assert upload_call_args.args[1] == output_bucket
    assert upload_call_args.args[2] == output_key


def test_redact_multi_page_text_pdf_with_pii(mock_s3_client, mock_text_analyzer, mock_anonymizer_engine):
    """Test redacting a multi-page TEXT PDF with PII."""
    input_bucket = "test-text-bucket"
    input_key = "text_pdf_multi_page_pii.pdf" # 2 pages
    output_bucket = "output-text-bucket"
    output_key = "redacted_text_pdf_multi_page_pii.pdf"

    # Page 1: "Page 1: My name is Bob Smith." -> PII: "Bob Smith" (PERSON)
    # Page 2: "Page 2: My email is bob.smith@example.com and phone is 1234567890" -> PII: "bob.smith@example.com" (EMAIL_ADDRESS), "1234567890" (PHONE_NUMBER)
    results_page1 = [RecognizerResult(entity_type="PERSON", start=19, end=28, score=0.9)] # "Bob Smith"
    results_page2 = [
        RecognizerResult(entity_type="EMAIL_ADDRESS", start=18, end=40, score=0.95), # "bob.smith@example.com"
        RecognizerResult(entity_type="PHONE_NUMBER", start=52, end=62, score=0.88)   # "1234567890"
    ]
    mock_text_analyzer.analyze.side_effect = [results_page1, results_page2]

    result_key = redact_text_pdf(
        input_s3_bucket_name=input_bucket,
        input_s3_key_name=input_key,
        output_s3_bucket_name=output_bucket,
        output_s3_key_name=output_key,
        s3_client=mock_s3_client,
        analyzer=mock_text_analyzer,
        anonymizer=mock_anonymizer_engine,
        min_score_threshold=0.85,
        pii_entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
    )

    assert result_key == output_key
    assert mock_s3_client.download_file.call_count == 1
    assert mock_text_analyzer.analyze.call_count == 2
    
    # Check calls to analyzer for each page
    first_call_kwargs = mock_text_analyzer.analyze.call_args_list[0].kwargs
    assert first_call_kwargs['text'] == "Page 1: My name is Bob Smith.\n" # Added newline
    assert first_call_kwargs['entities'] == ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]

    second_call_kwargs = mock_text_analyzer.analyze.call_args_list[1].kwargs
    assert second_call_kwargs['text'] == "Page 2: My email is bob.smith@example.com and phone is 1234567890\n" # Added newline
    assert second_call_kwargs['entities'] == ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]

    assert mock_anonymizer_engine.anonymize.call_count == 2 # Called for each page with PII
    assert mock_s3_client.upload_fileobj.call_count == 1
    upload_call_args = mock_s3_client.upload_fileobj.call_args
    assert upload_call_args.args[1] == output_bucket
    assert upload_call_args.args[2] == output_key


def test_redact_text_pdf_no_pii(mock_s3_client, mock_text_analyzer, mock_anonymizer_engine):
    """Test redacting a TEXT PDF with no PII."""
    input_bucket = "test-text-bucket"
    input_key = "text_pdf_single_page_no_pii.pdf"
    output_bucket = "output-text-bucket"
    output_key = "redacted_text_pdf_no_pii.pdf"

    mock_text_analyzer.analyze.return_value = [] # No PII found

    result_key = redact_text_pdf(
        input_s3_bucket_name=input_bucket,
        input_s3_key_name=input_key,
        output_s3_bucket_name=output_bucket,
        output_s3_key_name=output_key,
        s3_client=mock_s3_client,
        analyzer=mock_text_analyzer,
        anonymizer=mock_anonymizer_engine,
        min_score_threshold=0.9
    )

    assert result_key == output_key
    mock_s3_client.download_file.assert_called_once()
    mock_text_analyzer.analyze.assert_called_once() # Called for the single page
    # Anonymizer should not be called if no PII results from analyzer
    mock_anonymizer_engine.anonymize.assert_not_called()
    mock_s3_client.upload_fileobj.assert_called_once() # Still uploads the (unmodified) file


def test_redact_multi_page_text_pdf_no_pii(mock_s3_client, mock_text_analyzer, mock_anonymizer_engine):
    """Test redacting a multi-page TEXT PDF with no PII."""
    input_bucket = "test-text-bucket"
    input_key = "text_pdf_multi_page_no_pii.pdf" # 2 pages
    output_bucket = "output-text-bucket"
    output_key = "redacted_text_pdf_multi_page_no_pii.pdf"

    mock_text_analyzer.analyze.return_value = [] # No PII on any page

    result_key = redact_text_pdf(
        input_s3_bucket_name=input_bucket,
        input_s3_key_name=input_key,
        output_s3_bucket_name=output_bucket,
        output_s3_key_name=output_key,
        s3_client=mock_s3_client,
        analyzer=mock_text_analyzer,
        anonymizer=mock_anonymizer_engine
    )

    assert result_key == output_key
    assert mock_s3_client.download_file.call_count == 1
    assert mock_text_analyzer.analyze.call_count == 2 # Called for each page
    mock_anonymizer_engine.anonymize.assert_not_called()
    assert mock_s3_client.upload_fileobj.call_count == 1


# --- Test Cases for choose_and_run_redactor ---

@patch('file_redaction.redact_text_pdf')
@patch('file_redaction.redact_image_pdf')
def test_choose_image_redaction(mock_redact_image, mock_redact_text, mock_s3_client, mock_image_analyzer, mock_text_analyzer, mock_anonymizer_engine):
    """Test choosing and running image-based redaction."""
    input_bucket = "test-bucket"
    input_key = "image_input.pdf"
    output_bucket = "output-bucket"
    output_key = "redacted_image_output.pdf"
    expected_result_key = "mock_image_redaction_output_key"

    mock_redact_image.return_value = expected_result_key

    args = {
        "input_s3_bucket_name": input_bucket,
        "input_s3_key_name": input_key,
        "output_s3_bucket_name": output_bucket,
        "output_s3_key_name": output_key,
        "s3_client": mock_s3_client,
        "image_analyzer_engine": mock_image_analyzer, # For image redaction
        "analyzer": mock_text_analyzer,             # For text redaction (Presidio)
        "anonymizer": mock_anonymizer_engine,       # For text redaction (Presidio)
        "min_score_threshold": 0.75,
        "pii_entities": ["SSN"],
        "redaction_type": "image"
    }

    result = choose_and_run_redactor(**args)

    assert result == expected_result_key
    mock_redact_image.assert_called_once_with(
        input_s3_bucket_name=input_bucket,
        input_s3_key_name=input_key,
        output_s3_bucket_name=output_bucket,
        output_s3_key_name=output_key,
        s3_client=mock_s3_client,
        image_analyzer_engine=mock_image_analyzer,
        min_score_threshold=0.75,
        pii_entities=["SSN"]
    )
    mock_redact_text.assert_not_called()

@patch('file_redaction.redact_text_pdf')
@patch('file_redaction.redact_image_pdf')
def test_choose_text_redaction(mock_redact_image, mock_redact_text, mock_s3_client, mock_image_analyzer, mock_text_analyzer, mock_anonymizer_engine):
    """Test choosing and running text-based redaction."""
    input_bucket = "test-bucket"
    input_key = "text_input.pdf"
    output_bucket = "output-bucket"
    output_key = "redacted_text_output.pdf"
    expected_result_key = "mock_text_redaction_output_key"

    mock_redact_text.return_value = expected_result_key

    args = {
        "input_s3_bucket_name": input_bucket,
        "input_s3_key_name": input_key,
        "output_s3_bucket_name": output_bucket,
        "output_s3_key_name": output_key,
        "s3_client": mock_s3_client,
        "image_analyzer_engine": mock_image_analyzer, 
        "analyzer": mock_text_analyzer,      
        "anonymizer": mock_anonymizer_engine, 
        "min_score_threshold": 0.6,
        "pii_entities": ["EMAIL_ADDRESS"],
        "redaction_type": "text"
    }

    result = choose_and_run_redactor(**args)

    assert result == expected_result_key
    mock_redact_text.assert_called_once_with(
        input_s3_bucket_name=input_bucket,
        input_s3_key_name=input_key,
        output_s3_bucket_name=output_bucket,
        output_s3_key_name=output_key,
        s3_client=mock_s3_client,
        analyzer=mock_text_analyzer,
        anonymizer=mock_anonymizer_engine,
        min_score_threshold=0.6,
        pii_entities=["EMAIL_ADDRESS"],
        language="en" # Added language to expected call
    )
    mock_redact_image.assert_not_called()

def test_choose_invalid_redaction_type(mock_s3_client): # Only need s3_client for basic args
    """Test choosing an invalid redaction type."""
    args = {
        "input_s3_bucket_name": "test", "input_s3_key_name": "test.pdf",
        "output_s3_bucket_name": "out", "output_s3_key_name": "out.pdf",
        "s3_client": mock_s3_client, "redaction_type": "unknown",
        # Other args like analyzers, threshold, entities are not strictly needed
        # if the type check happens first.
        "image_analyzer_engine": None, "analyzer": None, "anonymizer": None,
        "min_score_threshold": 0.5, "pii_entities": ["ALL"]
    }
    with pytest.raises(ValueError, match="Invalid redaction_type: unknown. Must be 'image' or 'text'."):
        choose_and_run_redactor(**args)

# pytest tests/test_file_redaction.py

# Note: The `file_redaction.py` module is assumed to be in the parent directory
# or a location findable by Python's import system (e.g., by adding to PYTHONPATH
# or installing the package if it's structured as one).
# For this example, let's assume `file_redaction.py` is in `/app` and `tests` is in `/app/tests`.
# We might need to adjust sys.path for the import to work if running directly.
# However, pytest usually handles this well if run from the root of the project.

# Create a dummy file_redaction.py for the tests to run against
# This is a placeholder for the actual implementation.
# Ensure this path is correct relative to where pytest will be run.
# If tests are in /app/tests and file_redaction.py is in /app:
# .. file_redaction.py
# For now, I'll assume pytest is run from /app, so `from file_redaction import ...` works if `file_redaction.py` is in `/app`.

# A simple placeholder for redact_image_pdf in file_redaction.py might look like:
# (This is just for the test structure to have something to import)
"""
# file_redaction.py (dummy version for testing structure)
import fitz # PyMuPDF
from PIL import Image
import io
import os
# from comprehend.image_comprehend_parser import CustomImageAnalyzerEngine # Actual import

class CustomImageAnalyzerEngine: # Dummy for import
    def __init__(self, supported_entities=None):
        pass
    def analyze_image(self, image_bytes, entities):
        return []

def redact_image_pdf(input_s3_bucket_name, input_s3_key_name, output_s3_bucket_name, output_s3_key_name,
                     image_analyzer_engine, s3_client, min_score_threshold=0.5, pii_entities=["ALL"]):
    
    local_input_pdf_path = f"/tmp/{os.path.basename(input_s3_key_name)}"
    local_output_pdf_path = f"/tmp/redacted_{os.path.basename(input_s3_key_name)}"

    try:
        s3_client.download_file(input_s3_bucket_name, input_s3_key_name, local_input_pdf_path)
        
        doc = fitz.open(local_input_pdf_path)
        if not doc:
            raise Exception("Could not open PDF")

        total_pages = doc.page_count
        redacted_something = False

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png") # Get image bytes from page

            # Analyze image for PII
            pii_results = image_analyzer_engine.analyze_image(image_bytes=img_bytes, entities=pii_entities)
            
            img_width = pix.width
            img_height = pix.height

            for pii in pii_results:
                if pii.score >= min_score_threshold:
                    x0 = pii.left * img_width
                    y0 = pii.top * img_height
                    x1 = (pii.left + pii.width) * img_width
                    y1 = (pii.top + pii.height) * img_height
                    
                    page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(0,0,0)) # Black redaction
                    redacted_something = True
            
            if redacted_something: # Only apply if redactions were added for this page
                page.apply_redactions()
        
        if redacted_something or total_pages > 0 : # Save even if no redactions to ensure output exists
             doc.save(local_output_pdf_path)
        else: # If doc was empty and no redactions, what to do?
            # For now, let's assume we always want an output, even if it's a copy of input
            # Or if input was empty, an empty output.
            # The current mock s3_client.download always creates a file.
            # If the input PDF was truly empty or unreadable, doc.page_count might be 0.
            # Copy input to output if no pages or no redactions
             if os.path.exists(local_input_pdf_path) and not os.path.exists(local_output_pdf_path):
                doc.save(local_output_pdf_path) # Save a copy if no redactions

        doc.close()

        if os.path.exists(local_output_pdf_path):
            with open(local_output_pdf_path, "rb") as f:
                s3_client.upload_fileobj(f, output_s3_bucket_name, output_s3_key_name)
            return output_s3_key_name
        else:
            # This case should ideally not happen if download creates a file and we always save.
            # If the input PDF was empty and resulted in no pages, we should still produce an "empty" PDF.
            # For now, let's assume the original file gets "copied" by saving it.
            raise Exception("Output PDF not created, but no explicit error raised during redaction.")

    finally:
        if os.path.exists(local_input_pdf_path):
            os.remove(local_input_pdf_path)
        if os.path.exists(local_output_pdf_path):
            os.remove(local_output_pdf_path)
"""
pass
