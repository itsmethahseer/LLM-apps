example_json_structure = """
        {
          "invoices": [
      {
        "General_Fields": {
          "Invoice_Number": "string",
          "Date": "string",
          "Borrower_Name": "string",
          "Borrower_gst_no": "string",
          "End_Client_Name": "string",
          "End_Client_gst_no": "string",
          "Bank_Name": "string",
          "Account_Name": "string",
          "Account_Number": "string",
          "IFSC_Code": "string",
          "Service_or_Billing_Period": "string",
          "Taxable_Amount": "<value_without_symbol>",
          "Grand_Total": "<value_without_symbol>",
          "Purchase_Order_Number": "string"
        },
        "Line_Item_Details": [
        {
          "Serial_Number": "string",
          "Description": "string",
          "MRP": "string",
          "Discount_Percentage": "string",
          "Discount_Price": "string",
          "Tax_Percentage": "string",
          "HSN_or_SAC_Code": "string",
          "Quantity": "string",
          "Price" : "string",
          "Value" : "string"
        }
        ]
      }
    ]
        }
"""



common_prompt =f""" 
you have a list of tax invoices you want to act a information extractor from those invoices.    
Understand the invoice structure and accurately extract the following information:
                                **General_Fields**
                                Invoice_Number: "What is the unique identifier for this invoice?"
                                Date: "When was this invoice issued?"
                                Borrower_Name: "Who is the borrower or seller or biller of this invoice? Remove quotations, Place, floor etc if found along the borrower name. Do not remove titles if any like "M/S" , "M/R" , "LLP", etc. If same name is found 2 times nearby, only extract 1.
                                Borrower_gst_no: "What is the borrower's or seller's or biller's  GST number?"
                                End_Client_Name: "Who is the final client or buyer for this invoice?  Remove quotations, Place, floor etc if found along the client name. Do not remove titles if any like "M/S" , "M/R" , "LLP", etc."
                                End_Client_gst_no: "What is the final client's or buyer's GST number?"
                                Is_Bank_Details_Available?: "Is the bank details specified anywhere in the bill?"
                                Bank_Name: "Which bank is associated with this invoice? Only extract the name part after removing numbers if any."
                                Account_Name: "Who is the reciever for the payment? Most of the cases, it will be same as the borrowers name. If not found look for keywords like 'receiver', 'signatory', 'For' near Bank details."
                                Account_Number: "What is the bank account number for the payment?"
                                IFSC_Code: "What is the IFSC code of the bank?"
                                Service_or_Billing_Period: "What is the service or billing period covered by this invoice?"
                                Taxable_Amount: What is the total taxable amount in the invoice if possible, search the keywords like "Taxable Amt." else return as "NA".
                                Grand_Total: "What is the total amount due, including taxes?"
                                Purchase_Order_Number: "If applicable, what is the purchase order number associated with this invoice?"
                                Line_Item_Details
                                Serial_Number: What is the sequential number of this line item, it should be a `Serial_Number` please verify before extracting. if not specified take is as "NA".
                                Description_or_Nature_of_Service: What is the description of the product's line item? it should  alphabetical exclude numerical, if it is only numberical return as "NA".If not specified take it as "NA"
                                    -Example1 : ``` 
                                    Sr No  Product/description
                                    449    Venanta ROOHI 1290
                                    450    Vestoria
                                    ```
                                    -Example 2 : ```
                                    GST Summary

                                    S.N.	Description of Goods	HSN/SAC Code	Qty.
                                    39241090	12%				3.00 PCS	
                                    39249010	18%				4.00 PCS	
                                    39249090	18%				10.00 PCS	
                                    ```
                                    -Example3 : ```
                                    SI	Description of Goods
                                    475	Pip Squeak - 2s (Ref 2012)
                                    476	Animal Squeakers Pals 2pc
                                    477	Aqua Squeakers
                                    ```
                                    **If you see Line item in Example1 format you can directly takes the Serial_Number as "449" and "450" and `Description_or_Nature_of_Service` should be "Venanta ROOHI 1290" and "Vestoria"
                                    **If you see Line item in Example2 format you can directly takes the Serial_Number as "475", "476" and "477" and `Description_or_Nature_of_Service` should be "Pip Squeak - 2s (Ref 2012)", "Animal Squeakers Pals 2pc" and "Aqua Squeakers".
                                    ** But if you see Line item in Example2 format , it is not actually a product line item, it is GST tax summary , so `Serial_Number` should be "NA" and `Description_or_Nature_of_Service` also be "NA".
                                MRP: "What is the maximum retail price (MRP) of the item? No need of symbol."
                                Discount_Percentage: "What is the discount percentage applied to this line item?. if not specified take it as NA."
                                Discount_Price: "What is the discount amount calculated based on the MRP and discount percentage for this line item? if not specified take it as NA."
                                Tax_Percentage: "What is the tax percentage applicable to this individual line item?. If UGST, SGST/UTGST both are given for an item, then the Tax_Percentage will be their sum together. Take as 'NA' if not found individually for each item."
                                HSN_or_SAC_Code: "What is the Harmonized System of Nomenclature (HSN) or Standard Accounting Code (SAC) code for the item?"
                                Quantity: How many units of the item were provided. Extract the exact number in the invoice sample if possible else return as "NA".
                                Price: "What is the unit price of the item without the tax? No need of symbol."
                                Value: "What is the total value of this line item, including taxes and discounts? No need of symbol."

                                **Instructions for returning the data**
                                    One input may contain one or more than one invoice data. Identify the no. of unique Invoice_Number in the input to know the count.
                                    There is no relation between different invoices in an input, ie if for example if second last column of an invoice contains tax percentage, there is no rule that every invoice's second last column contains the same. Watch carefully the titles of the column.
                                    Do not miss any titles like 'LLP', 'M/S' etc from Borrower_Name and Client_Name. 
                                    If short name and full name both is seen as Borrower_Name and Client_Name , extract only the full name. For example for 'AS Allen Solly', only extract 'Allen Solly'.
                                    Do not add newline character between any values if found as multiple lines, instead add a white space.
                                    Do not get confused between 'MRP' and other rates. Only extract the MRP if the column 'MRP' is present, else extract as 'NA'.
                                    All the values should be in string format even if the value is numeric. Add % symbols to the values in percentage fields.
                                    If any of these fields are missing or cannot be accurately extracted, return "NA" for the corresponding field.
                                -Note : **extract every line item in the invoice without missing any.**
                                -Note : ** Please verify the Serial Number extracted is a valid Serial Number, if not return that as "NA".
                                 Please provide the results in the format as in the {example_json_structure}.Do not add any extra key in to the json.


"""

import os
import openai
from io import BytesIO
from PIL import Image
import fitz
import logging
from pixl_api_submodule import APP_CONFIG, ApiException
from typing import List, Dict
from PyPDF2 import PdfReader
import json
import base64
import google.generativeai as genai
from vertexai.preview import tokenization
from dotenv import load_dotenv
from decrypt import decrypt
load_dotenv(".env",override=True)
decrypt_key = os.getenv("API_KEY")
api_key=decrypt(decrypt_key)
import asyncio

model_name = "gemini-1.5-pro"
tokenizer = tokenization.get_tokenizer_for_model(model_name)
system_instruction = "You are a skilled Invoice Parsing Expert. Analyze the base64 encoded pdf given and extract the required details according to the instructions."
generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
            top_p=0.01,
        )
model = genai.GenerativeModel('gemini-1.5-pro',system_instruction=system_instruction,generation_config=generation_config)

def pdf_to_images(pdf_bytes):
    # Open the PDF file from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(pdf_document)
    logging.info(f"PDF document opened successfully. Page count: {page_count}.")
    # List to store images
    images = []
    
    # Iterate through each page
    for page_number in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_number)
        
        # Convert the page to a pixmap (image)
        pix = page.get_pixmap(dpi=300)
        
        # Convert the pixmap to an image (PNG format)
        image = pix.tobytes("png")
        
        # Append the image to the list
        images.append(image)
    
    return page_count,images

def extract_text_from_data_uri(data_uri, dpi=300):
    def base64_to_pdf(base64_data):
        logging.info("Converting base64 to PDF bytes.")
        pdf_bytes = base64.b64decode(base64_data)
        pdf_file = BytesIO(pdf_bytes)
        return pdf_file
    try:
        if data_uri.startswith('data:application/pdf'):
            pdf_bytes = base64_to_pdf(data_uri.split(',', 1)[1])
            page_count, images = pdf_to_images(pdf_bytes)
        else:
            raise ApiException("Unsupported data URI format")

    except Exception as e:
        logging.error(f"Error occurred during text extraction: {str(e)}")
        raise ApiException("Extraction and Processing Error", 500)
    return page_count, images

def process_json(invoices):
    max_taxable_amount = 'NA'
    max_grand_total = 'NA'

    for invoice in invoices:
        general_fields = invoice['General_Fields']
        taxable_amount = general_fields.get('Taxable_Amount', 'NA')
        grand_total = general_fields.get('Grand_Total', 'NA')

        if taxable_amount!= 'NA' and (max_taxable_amount == 'NA' or taxable_amount > max_taxable_amount):
            max_taxable_amount = taxable_amount
        if grand_total!= 'NA' and (max_grand_total == 'NA' or grand_total > max_grand_total):
            max_grand_total = grand_total

    general_fields = invoices[0]['General_Fields']
    general_fields['Taxable_Amount'] = max_taxable_amount
    general_fields['Grand_Total'] = max_grand_total

    processed_invoices = {
        'General_Fields': general_fields,
        'Line_Item_Details': []
    }

    for invoice in invoices:
        line_items = invoice['Line_Item_Details']
        processed_invoices['Line_Item_Details'].extend(line_items)

    return json.dumps(processed_invoices)


async def process_image(image: bytes, prompt: str) -> str:
    try:
        # Save the image to a temporary file
        image_file = Image.open(BytesIO(image))
        image_file.save("temp_image.jpeg")
        
        # Load the image file using PIL
        image = Image.open("temp_image.jpeg")
        
        # Convert the image to a bytes object
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        genai.configure(api_key=api_key)
 


        # Process the image using the Gemini model
        image_parts = [
            {
                "mime_type": "image/jpeg",  # Adjust mime type if needed
                "data": image_bytes.read()
            }
        ]
        
        response = await model.generate_content_async([prompt, image_parts[0]])
        output_text = response.text
        input_token_count = tokenizer.count_tokens(prompt).total_tokens
        output_token_count = tokenizer.count_tokens(output_text).total_tokens
        total_token_count = input_token_count + output_token_count
        logging.info(f"Total token count: {total_token_count:,}")
        # Return the response text
        return response.text
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise ApiException("Image Processing Error", 500)

async def extract_details_from_documents_async(base64data, **kwargs):
    """Processes images in batches asynchronously and extracts details."""
    logging.info("Extracting details from documents asynchronously.")
    page_count, images = extract_text_from_data_uri(base64data)
    chunk_size = 2  # Number of images per chunk
    combined_result = []  # To store final combined results

    # Split images into chunks
    image_chunks = list(chunk_list(images, chunk_size))
    logging.info(f"Images split into {len(image_chunks)} chunks of up to {chunk_size} images each.")

    # Create async tasks for each chunk
    tasks = []
    prompt = common_prompt
    for chunk in image_chunks:
        task = asyncio.create_task(process_chunk_async(chunk, prompt))
        tasks.append(task)

    # Process all tasks asynchronously in batches
    try:
        batch_results = await asyncio.gather(*tasks)
        combined_result.extend(batch_results)
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        raise ApiException("Batch Processing Error", 500)

    logging.info("All chunks processed successfully. Returning combined results.")
    final_results = {"invoices": combined_result}
    return page_count, final_results

async def process_chunk_async(chunk, prompt):
    """Processes a single chunk asynchronously."""
    try:
        logging.info("Processing a chunk asynchronously.")
        results = await asyncio.gather(*[process_image(image, prompt) for image in chunk])
        logging.info("Chunk processed successfully.")
        return results
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        raise ApiException("Chunk Processing Error", 500)

def chunk_list(lst, chunk_size):
    """Splits a list into smaller chunks."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]








        