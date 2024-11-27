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
        pix = page.get_pixmap(dpi=600)
        
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

# Prepare the images for the API call
def save_image(image, file_name: str = "temp_image.jpeg"):
    """Saves an image to a temporary file."""
    image = Image.open(BytesIO(image))
    image.save(file_name)
    logging.info(f"Image saved to {file_name} successfully.")

def encode_image_to_base64(file_name: str = "temp_image.jpeg") -> str:
    """Encodes an image file to a base64 string."""
    with open(file_name, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    logging.info(f"Image {file_name} encoded to base64 successfully.")
    return image_base64

def create_image_messages(images: List) -> List[Dict]:
    """Creates a list of image messages with base64 encoded images."""
    logging.info("Creating image messages for API call.")
    image_messages = []
    for image in images:
        save_image(image)
        image_base64 = encode_image_to_base64()
        image_messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        })
    logging.info("Image messages created successfully.")
    return image_messages

def create_text_message(text: str) -> Dict:
    """Creates a text message dictionary."""
    return {"type": "text", "text": text}

def create_request_payload(text, images: List) -> Dict:
    """Creates the payload for the OpenAI API request."""
    logging.info("Creating request payload for OpenAI API.")
    image_messages = create_image_messages(images)
    messages = [
        {
            "role": "user",
            "content": [
                create_text_message(text),
            ] + image_messages
        }
    ]
    logging.info("Request payload created successfully.")
    return messages

async def process_chunk_async(chunk, common_prompt):
    """Processes a single chunk asynchronously."""
    try:
        logging.info("Processing a chunk asynchronously.")
        messages = create_request_payload(common_prompt, chunk)
        response = await get_completion_async(messages)
        logging.info("Chunk processed successfully.")
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error: {str(e)}")
        raise ApiException("JSON Parsing Error", 500)
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        raise ApiException("Processing Error", 500)
    
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
    tasks = [process_chunk_async(chunk, common_prompt) for chunk in image_chunks]

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

async def get_completion_async(messages: List[Dict], model: str = 'gpt-4o', max_tokens: int = 16380) -> Dict:
    """Sends an asynchronous request to the OpenAI API and returns the response."""
    try:
        logging.info(f"Sending async request to OpenAI API with model: {model}.")
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            temperature=0,
            top_p=0.01,
            seed=0
        )
        logging.info("Received response from OpenAI API.")
        return response
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise ApiException("LLM call error")

def get_pdf_page_count(pdf_file_path: str) -> int:
    """Returns the number of pages in a PDF document."""
    reader = PdfReader(pdf_file_path)
    return len(reader.pages)

def chunk_list(lst, chunk_size):
    """Splits a list into smaller chunks."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def extract_details_from_documents(base64data, **kwargs):
    """Processes images and sends requests in chunks to extract details."""
    logging.info("Extracting details from documents.")
    page_count, images = extract_text_from_data_uri(base64data)
    chunk_size = 2  # Number of images per chunk
    combined_result = []  # To store final combined results

    # Split images into chunks
    image_chunks = list(chunk_list(images, chunk_size))
    logging.info(f"Images split into {len(image_chunks)} chunks of up to {chunk_size} images each.")

    for index, chunk in enumerate(image_chunks):
        logging.info(f"Processing chunk {index + 1} of {len(image_chunks)}.")
        # Create messages for the current chunk
        messages = create_request_payload(common_prompt, chunk)
        # Get response from the OpenAI API
        response = get_completion_async(messages)
        print("response",response.choices[0].message.content)
        # Extract and parse the content from the API response
        try:
            content = response.choices[0].message.content
            chunk_result = json.loads(content)
            combined_result.append(chunk_result)  # Append chunk results to combined result
            logging.info(f"Chunk {index + 1} processed successfully.")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error in chunk {index + 1}: {str(e)}")
            raise ApiException("JSON Parsing Error", 500)
        except Exception as e:
            logging.error(f"Error processing chunk {index + 1}: {str(e)}")
            raise ApiException("Processing Error", 500)
    print("combined result",combined_result)
    with open("hks.txt","w") as file:
        file.write(str(combined_result))
    logging.info("All chunks processed successfully. Returning combined results.")
    final_results = {"invoices": combined_result}
    return page_count, final_results