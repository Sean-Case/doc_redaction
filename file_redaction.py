import fitz # PyMuPDF
from PIL import Image, ImageDraw # Added ImageDraw for the dummy implementation
import io
import os

# Comprehend and other image-related imports/classes
# from comprehend.image_comprehend_parser import CustomImageAnalyzerEngine # Actual import

class CustomImageAnalyzerEngine: # Dummy for redact_image_pdf
    def __init__(self, supported_entities=None):
        self.supported_entities = supported_entities or []
    def analyze_image(self, image_bytes, entities):
        # This is a placeholder. The tests will mock this method directly.
        # print(f"Dummy CustomImageAnalyzerEngine.analyze_image called with entities: {entities}")
        return []

# Presidio related imports/classes
from presidio_analyzer import AnalyzerEngine, RecognizerResult # Actual imports for redact_text_pdf
from presidio_anonymizer import AnonymizerEngine

# Attempt to import Presidio entities, with fallback to dummy classes for problematic imports
try:
    from presidio_anonymizer.entities import AnonymizerRequest, EngineResult
except ImportError:
    print("WARNING: Failed to import AnonymizerRequest, EngineResult from presidio_anonymizer.entities. Using dummy classes.")
    class AnonymizerRequest:
        def __init__(self, text: str, analyzer_results: list, operators: dict = None):
            self.text = text
            self.analyzer_results = analyzer_results
            self.operators = operators if operators is not None else {}
            # print(f"Dummy AnonymizerRequest created with text: '{text[:30]}...', {len(analyzer_results)} results.")

    class EngineResult:
        def __init__(self, text: str = ""):
            self.text = text
            # print(f"Dummy EngineResult created with text: '{text[:30]}...'")

# Dummy AnalyzerEngine if needed (though tests mock the instance passed to redact_text_pdf)
# class AnalyzerEngine:
#     def __init__(self): pass
#     def analyze(self, text, language, entities, score_threshold): return []
# class AnonymizerEngine: # Dummy for type hint if needed, but tests mock the instance
#     def __init__(self): pass
#     def anonymize(self, request): return EngineResult(text="[DUMMY ANONYMIZED TEXT FROM DUMMY ENGINE]")


def redact_image_pdf(input_s3_bucket_name, input_s3_key_name, output_s3_bucket_name, output_s3_key_name,
                     image_analyzer_engine, s3_client, min_score_threshold=0.5, pii_entities=["ALL"]):
    
    base_input_key = os.path.basename(input_s3_key_name)
    local_input_pdf_path = f"/tmp/{base_input_key}"
    local_output_pdf_path = f"/tmp/redacted_{base_input_key}"

    try:
        print(f"Downloading {input_s3_key_name} from {input_s3_bucket_name} to {local_input_pdf_path}")
        s3_client.download_file(input_s3_bucket_name, input_s3_key_name, local_input_pdf_path)
        
        print(f"Opening PDF: {local_input_pdf_path}")
        doc = fitz.open(local_input_pdf_path)
        if not doc: # fitz.open raises an exception on failure, so this check might be redundant
            print("Error: Could not open PDF document.")
            raise Exception("Could not open PDF document.")

        total_pages = doc.page_count
        print(f"PDF has {total_pages} pages.")
        
        # This flag will track if any redactions were actually made across all pages.
        any_redactions_applied_to_pdf = False

        for page_num in range(total_pages):
            print(f"Processing page {page_num + 1}/{total_pages}")
            page = doc.load_page(page_num)
            
            # Convert page to image (PNG format)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")

            # Analyze image for PII
            # PII entities to search for, default to "ALL" if not specified or empty
            current_pii_entities = pii_entities if pii_entities else ["ALL"]
            print(f"Analyzing image for PII entities: {current_pii_entities} with threshold: {min_score_threshold}")
            pii_results = image_analyzer_engine.analyze_image(image_bytes=img_bytes, entities=current_pii_entities)
            
            img_width = pix.width
            img_height = pix.height
            print(f"Image dimensions: {img_width}x{img_height}. Found {len(pii_results)} potential PIIs before filtering.")

            page_had_redactions = False
            for pii in pii_results:
                print(f"PII: Type={pii.entity_type}, Score={pii.score}, Box=[{pii.left},{pii.top},{pii.width},{pii.height}]")
                if pii.score >= min_score_threshold:
                    # Define the redaction rectangle based on PII coordinates
                    # Coordinates from analyzer are relative (0.0-1.0), convert to absolute points for fitz
                    x0 = pii.left * img_width
                    y0 = pii.top * img_height
                    x1 = (pii.left + pii.width) * img_width
                    y1 = (pii.top + pii.height) * img_height
                    
                    redaction_rect = fitz.Rect(x0, y0, x1, y1)
                    print(f"Adding redaction annotation for {pii.entity_type} at {redaction_rect}")
                    page.add_redact_annot(redaction_rect, fill=(0, 0, 0)) # Black fill for redaction
                    page_had_redactions = True
                    any_redactions_applied_to_pdf = True
            
            if page_had_redactions:
                print(f"Applying redactions to page {page_num + 1}")
                page.apply_redactions()
            else:
                print(f"No redactions applied to page {page_num + 1} (either no PII or below threshold).")
        
        # Save the modified PDF (or original if no redactions)
        # Always save the document to ensure an output file is created.
        # If no redactions were made, it saves the original content to the new output path.
        print(f"Saving processed PDF to {local_output_pdf_path}. Any redactions applied: {any_redactions_applied_to_pdf}")
        doc.save(local_output_pdf_path)
        doc.close() # Close the document after saving

        # Upload the processed PDF to S3
        if os.path.exists(local_output_pdf_path):
            print(f"Uploading {local_output_pdf_path} to S3 bucket {output_s3_bucket_name} as {output_s3_key_name}")
            with open(local_output_pdf_path, "rb") as f:
                s3_client.upload_fileobj(f, output_s3_bucket_name, output_s3_key_name)
            print("Upload successful.")
            return output_s3_key_name
        else:
            # This case should ideally not be reached if doc.save() is always called.
            print("Error: Output PDF file not found locally after processing.")
            raise Exception("Output PDF not created locally, cannot upload to S3.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Re-raise the exception to allow test framework to catch it or for caller to handle
        raise
    finally:
        # Cleanup local temporary files
        if os.path.exists(local_input_pdf_path):
            # print(f"Cleaning up temporary input file: {local_input_pdf_path}")
            os.remove(local_input_pdf_path)
        if os.path.exists(local_output_pdf_path):
            # print(f"Cleaning up temporary output file: {local_output_pdf_path}")
            os.remove(local_output_pdf_path)


def redact_text_pdf(input_s3_bucket_name, input_s3_key_name, output_s3_bucket_name, output_s3_key_name,
                    s3_client, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine,
                    min_score_threshold=0.5, pii_entities=None, language="en"):
    
    base_input_key = os.path.basename(input_s3_key_name)
    local_input_pdf_path = f"/tmp/{base_input_key}"
    local_output_pdf_path = f"/tmp/redacted_text_{base_input_key}"

    if pii_entities is None:
        pii_entities = ["ALL"] # Default to all if not specified

    try:
        # print(f"Downloading {input_s3_key_name} from {input_s3_bucket_name} for text redaction.")
        s3_client.download_file(input_s3_bucket_name, input_s3_key_name, local_input_pdf_path)
        
        doc = fitz.open(local_input_pdf_path)
        if not doc:
            raise Exception("Could not open PDF for text redaction.")

        # print(f"PDF has {doc.page_count} pages for text redaction.")
        any_redactions_made = False

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text("text") # Extract text from page
            
            if not page_text.strip():
                # print(f"Page {page_num + 1} has no text content.")
                continue

            # print(f"Analyzing text for page {page_num + 1} with entities: {pii_entities}, threshold: {min_score_threshold}")
            # Note: presidio_analyzer.AnalyzerEngine.analyze takes 'entities' and 'score_threshold'
            analyzer_results = analyzer.analyze(text=page_text,
                                                language=language,
                                                entities=pii_entities,
                                                score_threshold=min_score_threshold)
            
            # print(f"Found {len(analyzer_results)} PII entities on page {page_num + 1} before filtering by min_score_threshold.")
            
            filtered_results = [r for r in analyzer_results if r.score >= min_score_threshold]


            if filtered_results:
                # print(f"Anonymizing page {page_num + 1} text with {len(filtered_results)} PII entities.")
                # Presidio AnonymizerEngine expects an AnonymizerRequest object
                # Using direct import for AnonymizerRequest:
                anonymizer_request_obj = AnonymizerRequest(
                    text=page_text,
                    analyzer_results=filtered_results,
                    operators={"DEFAULT": None} # Use default redaction or configure operators
                )
                # The anonymize method itself might take AnonymizerRequest directly or unpack kwargs.
                # Based on docs, it should take the request object.
                anonymized_result = anonymizer.anonymize(request=anonymizer_request_obj)
                
                # Redact the original page content. This is the tricky part.
                # We need to find the bounding boxes of the PII text and add redaction annotations.
                # PyMuPDF's search_for can find text and give its bounding box.
                
                # For each piece of PII found and anonymized by Presidio:
                # We need its original location (start, end in `page_text`)
                # Then find this text on the PDF page to get its bbox.
                
                # This simplified version will use page.add_redact_annot with search results
                # This is a common approach: search for the PII text segments from RecognizerResult
                # and then redact their bounding boxes.
                
                # NOTE: This redaction logic is basic. True redaction requires careful handling
                # of text flow, multiple instances, and ensuring the anonymized_result.text
                # (which has placeholders like <PERSON>) maps correctly to redaction areas.
                # A more robust way is to use the `start` and `end` from RecognizerResult
                # to find text blocks in PyMuPDF and redact those.
                
                redactions_on_this_page = 0
                for pii_item in filtered_results:
                    # pii_item.start and pii_item.end give the character offsets in `page_text`
                    # We need to find all occurrences of this text segment on the page.
                    text_to_redact = page_text[pii_item.start:pii_item.end]
                    if not text_to_redact.strip(): continue

                    # print(f"Searching for '{text_to_redact}' on page {page_num +1} for redaction.")
                    # PyMuPDF's search_for returns a list of Rects. 'hit_max' is not a valid parameter.
                    text_instances = page.search_for(text_to_redact) 
                    
                    if text_instances:
                        for inst_rect in text_instances:
                            page.add_redact_annot(inst_rect, fill=(0,0,0)) # Black redaction
                            redactions_on_this_page +=1
                
                if redactions_on_this_page > 0:
                    page.apply_redactions()
                    any_redactions_made = True
                # print(f"Applied {redactions_on_this_page} redaction annotations to page {page_num+1}")

        # Save the document if redactions were made or if it's just to produce an output
        # print(f"Saving text-redacted PDF to {local_output_pdf_path}. Any redactions made: {any_redactions_made}")
        doc.save(local_output_pdf_path)
        doc.close()

        if os.path.exists(local_output_pdf_path):
            # print(f"Uploading {local_output_pdf_path} to S3 bucket {output_s3_bucket_name} (text redacted).")
            with open(local_output_pdf_path, "rb") as f:
                s3_client.upload_fileobj(f, output_s3_bucket_name, output_s3_key_name)
            # print("Text redaction upload successful.")
            return output_s3_key_name
        else:
            raise Exception("Output PDF (text redacted) not created locally.")

    except Exception as e:
        # print(f"An error occurred during text redaction: {e}")
        raise
    finally:
        if os.path.exists(local_input_pdf_path):
            os.remove(local_input_pdf_path)
        if os.path.exists(local_output_pdf_path):
            os.remove(local_output_pdf_path)


# --- Main execution / Standalone Test (Optional) ---
# ... (previous main section content) ...

def choose_and_run_redactor(
    redaction_type: str,
    input_s3_bucket_name: str,
    input_s3_key_name: str,
    output_s3_bucket_name: str,
    output_s3_key_name: str,
    s3_client,
    image_analyzer_engine=None, # Specific to image redaction
    analyzer=None,              # Specific to text redaction (Presidio AnalyzerEngine)
    anonymizer=None,            # Specific to text redaction (Presidio AnonymizerEngine)
    min_score_threshold: float = 0.5,
    pii_entities: list = None,
    language: str = "en"        # Specific to text redaction
):
    """
    Chooses and runs the appropriate redaction function based on redaction_type.
    Forwards all relevant arguments to the chosen function.
    """
    if redaction_type == "image":
        if image_analyzer_engine is None:
            raise ValueError("image_analyzer_engine is required for 'image' redaction type.")
        # print(f"Choosing image redaction for {input_s3_key_name}")
        return redact_image_pdf(
            input_s3_bucket_name=input_s3_bucket_name,
            input_s3_key_name=input_s3_key_name,
            output_s3_bucket_name=output_s3_bucket_name,
            output_s3_key_name=output_s3_key_name,
            s3_client=s3_client,
            image_analyzer_engine=image_analyzer_engine,
            min_score_threshold=min_score_threshold,
            pii_entities=pii_entities if pii_entities else ["ALL"] 
        )
    elif redaction_type == "text":
        if analyzer is None or anonymizer is None:
            raise ValueError("analyzer and anonymizer are required for 'text' redaction type.")
        # print(f"Choosing text redaction for {input_s3_key_name}")
        return redact_text_pdf(
            input_s3_bucket_name=input_s3_bucket_name,
            input_s3_key_name=input_s3_key_name,
            output_s3_bucket_name=output_s3_bucket_name,
            output_s3_key_name=output_s3_key_name,
            s3_client=s3_client,
            analyzer=analyzer,
            anonymizer=anonymizer,
            min_score_threshold=min_score_threshold,
            pii_entities=pii_entities if pii_entities else ["ALL"],
            language=language
        )
    else:
        raise ValueError(f"Invalid redaction_type: {redaction_type}. Must be 'image' or 'text'.")

if __name__ == '__main__':
    print("file_redaction.py executed as main script.")
    # Example usage (requires setting up mock clients and engines):
    # pass
