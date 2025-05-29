import pytest
import pandas as pd
import numpy as np # For NaN values
from unittest.mock import MagicMock, patch, mock_open
from PIL import Image, ImageDraw # Added for mocking image properties
import xml.etree.ElementTree as ET # For mocking XFDF parsing
import fitz # PyMuPDF
from fitz import Document, Page, Rect # Specific imports from PyMuPDF
from datetime import datetime, timezone # For mocking datetime.now()
import gradio as gr # For gr.Progress

# Adjust this import based on your project structure
try:
    from ..tools.redaction_review import (
        decrease_page, 
        increase_page, 
        update_dropdown_list_based_on_dataframe,
        undo_last_removal,
        exclude_selected_items_from_redaction,
        replace_annotator_object_img_np_array_with_page_sizes_image_path,
        get_boxes_json,
        replace_placeholder_image_with_real_image,
        get_all_rows_with_same_text,
        update_selected_review_df_row_colour,
        update_boxes_color,
        fill_missing_ids, 
        convert_image_coords_to_adobe,
        convert_pymupdf_coords_to_adobe,
        parse_xfdf,
        convert_adobe_coords_to_image,
        create_xfdf,
        convert_df_to_xfdf,
        convert_xfdf_to_dataframe,
        detect_file_type, 
        get_file_name_without_type,
        is_pdf, 
        redact_page_with_pymupdf, 
        save_pdf_with_or_without_compression, 
        convert_annotation_json_to_review_df, 
        divide_coordinates_by_page_sizes, 
        apply_redactions_to_review_df_and_files,
    )
    from ..tools.file_conversion import multiply_coordinates_by_page_sizes 
    # from gradio_image_annotation.image_annotator import AnnotatedImageData # If needed
except ImportError:
    from tools.redaction_review import (
        decrease_page, 
        increase_page, 
        update_dropdown_list_based_on_dataframe,
        undo_last_removal,
        exclude_selected_items_from_redaction,
        replace_annotator_object_img_np_array_with_page_sizes_image_path,
        get_boxes_json,
        replace_placeholder_image_with_real_image,
        get_all_rows_with_same_text,
        update_selected_review_df_row_colour,
        update_boxes_color,
        fill_missing_ids,
        convert_image_coords_to_adobe,
        convert_pymupdf_coords_to_adobe,
        parse_xfdf,
        convert_adobe_coords_to_image,
        create_xfdf,
        convert_df_to_xfdf,
        convert_xfdf_to_dataframe,
        detect_file_type, 
        get_file_name_without_type,
        is_pdf,
        redact_page_with_pymupdf,
        save_pdf_with_or_without_compression,
        convert_annotation_json_to_review_df,
        divide_coordinates_by_page_sizes,
        apply_redactions_to_review_df_and_files,
    )
    from tools.file_conversion import multiply_coordinates_by_page_sizes
    # from gradio_image_annotation.image_annotator import AnnotatedImageData


# Default color and highlight color used by the function being tested
DEFAULT_COLOUR = (0,0,0) 
HIGHLIGHT_COLOUR = (255,0,0) 
BOX_HIGHLIGHT_COLOUR = 'red' 
BOX_DEFAULT_COLOUR = 'blue' 
XFDF_DEFAULT_COLOR = '(0,0,0)' 

# Mock for fill_missing_ids
def mock_fill_missing_ids(df):
    if df is None: return pd.DataFrame() 
    df_copy = df.copy()
    if 'id' not in df_copy.columns or df_copy['id'].isnull().any():
        df_copy['id'] = [f'mock_id_{i}' for i in range(len(df_copy))]
    return df_copy

# --- Existing tests (Ensure they are present) ---
# (Omitted for brevity, but they should all be here in the final file)
def test_decrease_page_valid(): assert decrease_page(5) == (4, 4)
def test_increase_page_valid():
    assert increase_page(1, [{'image': 'p1'}, {'image': 'p2'}]) == (2, 2)
# ... and so on for all previously added tests ...
@patch('tools.redaction_review.datetime') 
@patch('fitz.open') 
@patch('tools.redaction_review.multiply_coordinates_by_page_sizes') 
def test_create_xfdf_absolute_coords(mock_multiply_coords, mock_fitz_open, mock_datetime_utc):
    mock_now_utc = datetime(2023, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
    mock_datetime_utc.now.return_value = mock_now_utc
    expected_date_str = "20230101120000123Z"
    mock_pdf_doc = MagicMock(spec=Document); mock_fitz_open.return_value = mock_pdf_doc
    mock_page_obj = MagicMock(spec=Page); mock_pdf_doc.load_page.return_value = mock_page_obj
    page_rect = fitz.Rect(0, 0, 600, 800)
    mock_page_obj.mediabox = page_rect; mock_page_obj.cropbox = page_rect; mock_page_obj.rect = page_rect
    review_file_df = pd.DataFrame([{'page': 1, 'xmin': 50, 'ymin': 100, 'xmax': 150, 'ymax': 200, 'label': 'PERSON', 'text': 'John Doe', 'color': '(255,0,0)'}])
    xfdf_content = create_xfdf(review_file_df, "dummy.pdf", mock_pdf_doc, page_sizes_df=None, document_cropboxes=None)
    mock_multiply_coords.assert_not_called(); mock_pdf_doc.load_page.assert_called_once_with(0)
    root = ET.fromstring(xfdf_content); assert root.tag == 'xfdf'
    redact = root.find('annots/redact'); assert redact is not None
    assert redact.get('rect') == "50.0,600.0,150.0,700.0"; assert redact.get('color') == '#FF0000'

@patch('tools.redaction_review.datetime')
@patch('fitz.open')
@patch('tools.redaction_review.multiply_coordinates_by_page_sizes')
def test_create_xfdf_normalized_coords_with_cropbox(mock_multiply_coords, mock_fitz_open, mock_datetime_utc):
    mock_now_utc = datetime(2023, 10, 26, 10, 30, 0, 0, tzinfo=timezone.utc); mock_datetime_utc.now.return_value = mock_now_utc
    #expected_date_str = "20231026103000000Z" # Original line
    mock_pdf_doc = MagicMock(spec=Document); mock_fitz_open.return_value = mock_pdf_doc
    mock_page_obj = MagicMock(spec=Page); mock_pdf_doc.load_page.return_value = mock_page_obj
    mock_page_obj.mediabox = fitz.Rect(0, 0, 700, 1000)
    cropbox_rect = fitz.Rect(50, 150, 550, 850); mock_page_obj.cropbox = cropbox_rect; mock_page_obj.rect = cropbox_rect
    review_file_df_normalized = pd.DataFrame([{'page': 1, 'xmin': 0.1, 'ymin': 0.1, 'xmax': 0.2, 'ymax': 0.2, 'label': 'LOCATION', 'text': 'Some Place', 'color': '(0,0,255)'}])
    page_sizes_df = pd.DataFrame([{'page': 1, 'image_path': 'img_page1.png', 'image_width': 500, 'image_height': 700}])
    def custom_multiplier(df, ps_df, cols_to_multiply): # Corrected argument name
        df_out = df.copy(); img_dims = ps_df[ps_df['page'] == df_out.loc[0,'page']].iloc[0]
        for col in ['xmin','xmax']: df_out[col] *= img_dims['image_width']
        for col in ['ymin','ymax']: df_out[col] *= img_dims['image_height']
        return df_out
    mock_multiply_coords.side_effect = custom_multiplier
    document_cropboxes = {0: cropbox_rect}
    xfdf_content = create_xfdf(review_file_df_normalized, "cropped.pdf", mock_pdf_doc, page_sizes_df, document_cropboxes)
    mock_multiply_coords.assert_called_once()
    mock_page_obj.set_cropbox.assert_called_once_with(cropbox_rect)
    root = ET.fromstring(xfdf_content); redact = root.find('annots/redact'); assert redact is not None
    assert redact.get('rect') == "50.0,560.0,100.0,630.0"; assert redact.get('color') == '#0000FF'

@patch('builtins.open', new_callable=mock_open)
@patch('tools.redaction_review.create_xfdf')
@patch('pandas.read_csv')
@patch('fitz.open')
def test_convert_df_to_xfdf_valid(mock_fitz_pdf_open, mock_pd_read_csv, mock_rr_create_xfdf, mock_file_open):
    sample_df = pd.DataFrame([{'page': 1, 'text': 'data'}]); mock_pd_read_csv.return_value = sample_df
    mock_rr_create_xfdf.return_value = "<xfdf_mock_content />"
    mock_pdf_doc = MagicMock(spec=Document); mock_fitz_pdf_open.return_value.__enter__.return_value = mock_pdf_doc
    with patch('tools.redaction_review.detect_file_type', side_effect=lambda x: 'csv' if 'csv' in x else ('pdf' if 'pdf' in x else 'unknown')), \
         patch('tools.redaction_review.get_file_name_without_type', side_effect=lambda x: x.split('.')[0]):
        input_files = ['review.csv', 'document.pdf']; output_folder = "output/"
        result_paths = convert_df_to_xfdf(input_files, output_folder, None)
    mock_rr_create_xfdf.assert_called_once_with(sample_df, 'document.pdf', mock_pdf_doc, None, None)
    mock_file_open.assert_called_once_with("output/document.xfdf", 'w', encoding='utf-8')
    assert result_paths == ["output/document.xfdf"]

@patch('pandas.DataFrame.to_csv')
@patch('PIL.Image.open')
@patch('tools.redaction_review.convert_adobe_coords_to_image')
@patch('tools.redaction_review.parse_xfdf')
@patch('fitz.open')
def test_convert_xfdf_to_dataframe_valid(mock_fitz_open_pdf, mock_rr_parse_xfdf, mock_rr_convert_coords, mock_pil_img_open, mock_df_to_csv):
    mock_parsed_data = [{'page': 1, 'x1': 10, 'y1': 700, 'x2': 100, 'y2': 750, 'label': 'LBL1', 'text': 'TXT1', 'color': '(255,0,0)'}]
    mock_rr_parse_xfdf.return_value = mock_parsed_data
    mock_pdf_doc = MagicMock(spec=Document); mock_fitz_open_pdf.return_value.__enter__.return_value = mock_pdf_doc
    mock_page1 = MagicMock(spec=Page); mock_page1.rect = fitz.Rect(0,0, 600, 800); mock_pdf_doc.load_page.return_value = mock_page1
    mock_img1 = MagicMock(spec=Image.Image); mock_img1.size = (1200, 1600); mock_pil_img_open.return_value = mock_img1
    mock_rr_convert_coords.return_value = (20,100,200,200)
    with patch('tools.redaction_review.detect_file_type', side_effect=lambda x: 'xfdf' if 'xfdf' in x else ('pdf' if 'pdf' in x else ('png' if 'png' in x else 'unknown'))), \
         patch('tools.redaction_review.get_file_name_without_type', side_effect=lambda x: x.split('.')[0]):
        file_paths = ['input.xfdf', 'doc.pdf', 'page1_img.png']; output_folder = "output_csv/"
        result_paths = convert_xfdf_to_dataframe(file_paths, output_folder)
    mock_rr_convert_coords.assert_called_once_with(600.0, 800.0, 1200, 1600, 10, 700, 100, 750)
    mock_df_to_csv.assert_called_once_with("output_csv/doc_redactions.csv", index=False)
    assert result_paths == ["output_csv/doc_redactions.csv"]
# --- End of existing tests ---

# Helper to create a mock AnnotatedImageData-like object
def create_mock_annotated_image_data(page_num, boxes_data, image_path="dummy_image.png"):
    return {"image": image_path, "boxes": boxes_data, "page": page_num }

# Mock for gr.Progress
mock_progress_tracker = MagicMock(spec=gr.Progress)
# Ensure the mock_progress_tracker can be iterated over for track_tqdm
mock_progress_tracker.return_value = iter(range(100))


# --- Tests for apply_redactions_to_review_df_and_files ---
@patch('tools.redaction_review.get_file_name_without_type', return_value="test_image")
@patch('tools.redaction_review.is_pdf', return_value=False) 
@patch('tools.redaction_review.replace_annotator_object_img_np_array_with_page_sizes_image_path')
@patch('PIL.Image.open')
@patch('PIL.ImageDraw.Draw')
@patch('fitz.open') 
@patch('tools.redaction_review.redact_page_with_pymupdf')
@patch('tools.redaction_review.save_pdf_with_or_without_compression')
@patch('tools.redaction_review.convert_annotation_json_to_review_df')
@patch('tools.redaction_review.divide_coordinates_by_page_sizes')
@patch('pandas.DataFrame.to_csv') 
def test_apply_redactions_image_file(
    mock_df_to_csv, mock_divide_coords, mock_convert_json_to_df,
    mock_save_pdf, mock_redact_page_pymupdf, mock_fitz_open_call, 
    mock_pil_image_draw, mock_pil_image_open,
    mock_replace_annotator_path, mock_is_pdf_func, mock_get_name_func 
    ):

    mock_image_instance = MagicMock(spec=Image.Image)
    mock_image_instance.copy.return_value = mock_image_instance 
    mock_image_instance.convert.return_value = mock_image_instance 
    mock_pil_image_open.return_value = mock_image_instance
    mock_draw_instance = MagicMock(spec=ImageDraw.Draw)
    mock_pil_image_draw.return_value = mock_draw_instance

    file_paths = ["input/test_image.png"]
    doc = None 
    
    boxes_data = [{'xmin': 10, 'ymin': 10, 'xmax': 50, 'ymax': 50, 'label': 'REDACT', 'color': '(0,0,0)'}]
    all_image_annotations = [create_mock_annotated_image_data(page_num=1, boxes_data=boxes_data, image_path=file_paths[0])]
    
    current_page = 1 
    initial_review_df = pd.DataFrame([{'id': 'id1', 'page': 1, 'label': 'L1', 'text': 'T1'}]) 
    page_sizes = [{'page': 1, 'image_width': 100, 'image_height': 100, 'image_path': file_paths[0]}]
    output_folder = "redacted_output/"
    
    mock_replace_annotator_path.return_value = (all_image_annotations[0], all_image_annotations)
    updated_review_df_from_json = initial_review_df.copy(); updated_review_df_from_json['source'] = 'json_converted'
    mock_convert_json_to_df.return_value = updated_review_df_from_json
    final_review_df_for_csv = updated_review_df_from_json.copy(); final_review_df_for_csv['coords_divided'] = True
    mock_divide_coords.return_value = final_review_df_for_csv

    output_files, final_df_state = apply_redactions_to_review_df_and_files(
        doc, file_paths, all_image_annotations, current_page, 
        initial_review_df, page_sizes, output_folder, 
        save_pdf=True, progress=mock_progress_tracker 
    )

    mock_is_pdf_func.assert_called_once_with(file_paths[-1])
    mock_get_name_func.assert_called_with(file_paths[-1]) 
    mock_replace_annotator_path.assert_called_once_with(all_image_annotations, all_image_annotations[0], page_sizes, current_page)
    mock_pil_image_open.assert_called_once_with(file_paths[0])
    mock_draw_instance.rectangle.assert_called_once_with((10,10,50,50), fill=(0,0,0))
    mock_image_instance.save.assert_called_once_with("redacted_output/test_image_REDACTED.png")
    mock_fitz_open_call.assert_not_called()
    mock_convert_json_to_df.assert_called_once_with(all_image_annotations, page_sizes)
    pd.testing.assert_frame_equal(mock_df_to_csv.call_args[0][0], final_review_df_for_csv)
    assert output_files == ["redacted_output/test_image_REDACTED.png", "redacted_output/test_image_REDACTED.csv"]
    pd.testing.assert_frame_equal(final_df_state, final_review_df_for_csv)

@patch('tools.redaction_review.get_file_name_without_type', return_value="test_document")
@patch('tools.redaction_review.is_pdf', return_value=True) 
@patch('tools.redaction_review.replace_annotator_object_img_np_array_with_page_sizes_image_path') 
@patch('PIL.Image.open') 
@patch('PIL.ImageDraw.Draw') 
@patch('fitz.open') 
@patch('tools.redaction_review.redact_page_with_pymupdf')
@patch('tools.redaction_review.save_pdf_with_or_without_compression')
@patch('tools.redaction_review.convert_annotation_json_to_review_df')
@patch('tools.redaction_review.divide_coordinates_by_page_sizes')
@patch('pandas.DataFrame.to_csv')
def test_apply_redactions_pdf_file(
    mock_df_to_csv, mock_divide_coords, mock_convert_json_to_df,
    mock_save_pdf_func, mock_redact_page_func, mock_fitz_open_func,
    mock_pil_draw, mock_pil_open_func, 
    mock_replace_annotator, mock_is_pdf_func, mock_get_name_func
    ):

    file_paths = ["input/test_document.pdf"]
    mock_pdf_doc = MagicMock(spec=Document)
    mock_fitz_open_func.return_value.__enter__.return_value = mock_pdf_doc
    mock_pdf_doc.page_count = 2
    mock_page1, mock_page2 = MagicMock(spec=Page), MagicMock(spec=Page)
    mock_pdf_doc.load_page.side_effect = [mock_page1, mock_page2]

    boxes_p1 = [{'xmin':10,'ymin':10,'xmax':50,'ymax':50,'label':'L1','color':'(0,0,0)'}]
    boxes_p2 = [{'xmin':20,'ymin':20,'xmax':60,'ymax':60,'label':'L2','color':'(0,0,1)'}]
    all_ann = [create_mock_annotated_image_data(1,boxes_p1,"p1.png"), create_mock_annotated_image_data(2,boxes_p2,"p2.png")]
    initial_df = pd.DataFrame([{'id':'id1','page':1}]); page_sizes = [{'page':1}, {'page':2}] # Simplified page_sizes
    mock_convert_json_to_df.return_value = initial_df; mock_divide_coords.return_value = initial_df

    output_files, _ = apply_redactions_to_review_df_and_files(
        mock_pdf_doc, file_paths, all_ann, 1, initial_df, page_sizes, "out/", True, mock_progress_tracker
    )
    mock_is_pdf_func.assert_called_once_with(file_paths[-1])
    mock_fitz_open_func.assert_called_once_with(file_paths[-1])
    assert mock_redact_page_func.call_count == 2
    # Note: The actual page_size detail would be more complex if cropboxes were involved
    mock_redact_page_func.assert_any_call(mock_page1, boxes_p1, page_sizes[0], mock_pdf_doc, 0, None)
    mock_redact_page_func.assert_any_call(mock_page2, boxes_p2, page_sizes[1], mock_pdf_doc, 1, None)
    mock_save_pdf_func.assert_called_once_with(mock_pdf_doc, "out/test_document_REDACTED.pdf", compress_pdf=False)
    mock_df_to_csv.assert_called_once_with("out/test_document_REDACTED.csv", index=False)
    assert output_files == ["out/test_document_REDACTED.pdf", "out/test_document_REDACTED.csv"]

@patch('tools.redaction_review.get_file_name_without_type', return_value="test_doc_no_save")
@patch('tools.redaction_review.is_pdf', return_value=True)
@patch('tools.redaction_review.replace_annotator_object_img_np_array_with_page_sizes_image_path')
@patch('PIL.Image.open')
@patch('PIL.ImageDraw.Draw')
@patch('fitz.open')
@patch('tools.redaction_review.redact_page_with_pymupdf') 
@patch('tools.redaction_review.save_pdf_with_or_without_compression') 
@patch('tools.redaction_review.convert_annotation_json_to_review_df')
@patch('tools.redaction_review.divide_coordinates_by_page_sizes')
@patch('pandas.DataFrame.to_csv')
def test_apply_redactions_pdf_file_save_pdf_false(
    mock_df_to_csv, mock_divide_coords, mock_convert_json_to_df,
    mock_save_pdf_func, mock_redact_page_func, mock_fitz_open_func,
    mock_pil_draw, mock_pil_open_func, 
    mock_replace_annotator, mock_is_pdf_func, mock_get_name_func
    ):

    file_paths = ["input/test_doc_no_save.pdf"]
    mock_pdf_doc = MagicMock(spec=Document)
    mock_fitz_open_func.return_value.__enter__.return_value = mock_pdf_doc
    mock_pdf_doc.page_count = 1
    mock_page1 = MagicMock(spec=Page)
    mock_pdf_doc.load_page.return_value = mock_page1

    boxes_p1 = [{'xmin':10,'ymin':10,'xmax':50,'ymax':50,'label':'L1','color':'(0,0,0)'}]
    all_ann = [create_mock_annotated_image_data(1,boxes_p1,"p1.png")]
    initial_df = pd.DataFrame([{'id':'id1','page':1}]); page_sizes = [{'page':1}]
    mock_convert_json_to_df.return_value = initial_df; mock_divide_coords.return_value = initial_df

    output_files, _ = apply_redactions_to_review_df_and_files(
        mock_pdf_doc, file_paths, all_ann, 1, initial_df, page_sizes, "out/", 
        save_pdf=False, progress=mock_progress_tracker
    )
    
    mock_is_pdf_func.assert_called_once_with(file_paths[-1])
    mock_fitz_open_func.assert_called_once_with(file_paths[-1]) 
    mock_redact_page_func.assert_called_once() 
    mock_save_pdf_func.assert_not_called() 
    expected_csv_path = "out/test_doc_no_save_REDACTED.csv"
    mock_df_to_csv.assert_called_once_with(expected_csv_path, index=False)
    assert output_files == [file_paths[-1], expected_csv_path]

@patch('tools.redaction_review.get_file_name_without_type', return_value="test_image_empty_boxes")
@patch('tools.redaction_review.is_pdf', return_value=False)
@patch('tools.redaction_review.replace_annotator_object_img_np_array_with_page_sizes_image_path')
@patch('PIL.Image.open')
@patch('PIL.ImageDraw.Draw')
@patch('tools.redaction_review.convert_annotation_json_to_review_df')
@patch('tools.redaction_review.divide_coordinates_by_page_sizes')
@patch('pandas.DataFrame.to_csv')
def test_apply_redactions_empty_boxes(
    mock_df_to_csv, mock_divide_coords, mock_convert_json_to_df,
    mock_pil_image_draw, mock_pil_image_open,
    mock_replace_annotator_path, mock_is_pdf_func, mock_get_name_func
    ):

    mock_image_instance = MagicMock(spec=Image.Image); mock_image_instance.copy.return_value = mock_image_instance
    mock_image_instance.convert.return_value = mock_image_instance; mock_pil_image_open.return_value = mock_image_instance
    mock_draw_instance = MagicMock(spec=ImageDraw.Draw); mock_pil_image_draw.return_value = mock_draw_instance

    file_paths = ["input/test_image_empty_boxes.png"]
    all_image_annotations = [create_mock_annotated_image_data(page_num=1, boxes_data=[], image_path=file_paths[0])] # Empty boxes
    current_page = 1; initial_review_df = pd.DataFrame(); page_sizes = [{'page':1}]
    output_folder = "redacted_output/"
    
    mock_replace_annotator_path.return_value = (all_image_annotations[0], all_image_annotations)
    mock_convert_json_to_df.return_value = initial_review_df; mock_divide_coords.return_value = initial_review_df

    output_files, _ = apply_redactions_to_review_df_and_files(
        None, file_paths, all_image_annotations, current_page, 
        initial_review_df, page_sizes, output_folder, 
        save_pdf=True, progress=mock_progress_tracker
    )
    
    mock_pil_image_open.assert_called_once_with(file_paths[0])
    mock_draw_instance.rectangle.assert_not_called() # No boxes to draw
    mock_image_instance.save.assert_called_once_with("redacted_output/test_image_empty_boxes_REDACTED.png") # Still saves image
    expected_csv_path = "redacted_output/test_image_empty_boxes_REDACTED.csv"
    mock_df_to_csv.assert_called_once_with(expected_csv_path, index=False)
    assert output_files == ["redacted_output/test_image_empty_boxes_REDACTED.png", expected_csv_path]

@patch('tools.redaction_review.get_file_name_without_type', return_value="test_image_str_path")
@patch('tools.redaction_review.is_pdf', return_value=False)
@patch('tools.redaction_review.replace_annotator_object_img_np_array_with_page_sizes_image_path')
@patch('PIL.Image.open')
@patch('PIL.ImageDraw.Draw')
@patch('tools.redaction_review.convert_annotation_json_to_review_df')
@patch('tools.redaction_review.divide_coordinates_by_page_sizes')
@patch('pandas.DataFrame.to_csv')
def test_apply_redactions_filepath_is_string(
    mock_df_to_csv, mock_divide_coords, mock_convert_json_to_df,
    mock_pil_image_draw, mock_pil_image_open,
    mock_replace_annotator_path, mock_is_pdf_func, mock_get_name_func
    ):
    mock_image_instance = MagicMock(spec=Image.Image); mock_image_instance.copy.return_value = mock_image_instance
    mock_image_instance.convert.return_value = mock_image_instance; mock_pil_image_open.return_value = mock_image_instance

    file_path_str = "input/test_image_str_path.png" # String instead of list
    boxes_data = [{'xmin': 5, 'ymin': 5, 'xmax': 25, 'ymax': 25, 'label': 'REDACT', 'color': '(0,0,0)'}]
    all_image_annotations = [create_mock_annotated_image_data(page_num=1, boxes_data=boxes_data, image_path=file_path_str)]
    current_page = 1; initial_review_df = pd.DataFrame(); page_sizes = [{'page':1}]
    output_folder = "redacted_output/"

    mock_replace_annotator_path.return_value = (all_image_annotations[0], all_image_annotations)
    mock_convert_json_to_df.return_value = initial_review_df; mock_divide_coords.return_value = initial_review_df

    output_files, _ = apply_redactions_to_review_df_and_files(
        None, file_path_str, all_image_annotations, current_page, 
        initial_review_df, page_sizes, output_folder, 
        save_pdf=True, progress=mock_progress_tracker
    )
    
    mock_is_pdf_func.assert_called_once_with(file_path_str) # Called with the string path
    mock_get_name_func.assert_called_with(file_path_str)
    mock_pil_image_open.assert_called_once_with(file_path_str)
    mock_pil_image_draw.return_value.rectangle.assert_called_once()
    mock_image_instance.save.assert_called_once_with("redacted_output/test_image_str_path_REDACTED.png")
    expected_csv_path = "redacted_output/test_image_str_path_REDACTED.csv"
    mock_df_to_csv.assert_called_once_with(expected_csv_path, index=False)
    assert output_files == ["redacted_output/test_image_str_path_REDACTED.png", expected_csv_path]
