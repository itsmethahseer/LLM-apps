
from API.core.pipeline import extract_details_from_documents_async,process_json
@router.post(f"/extractor_updated", response_model=ApiResponse, description="optional description about endpoint", tags=["ServiceEndpoints"])
async def extractor(request:Request,input_json : parser, client_detail = Depends(check_authentication_header)):
    # start_time = time.time()
    data_ = input_json.input
    # Call the asynchronous extraction function
    page_count, result_ = await extract_details_from_documents_async(data_)
    print("type of response",type(result_))
    # print("response before processing",result_)
    # with open("results.txt","w") as file:
    #     file.write(str(result_))
    # result_ = process_json(result_)
    # Update request state for logging and tracking
    request.state.num_of_execution = 1
    request.state.num_of_pages = page_count
    request.state.client_reference_id = input_json.uuid
    request.state.remarks = "API success"
    request.state.custom_remarks = {}
    # Return the response
    return ApiResponse(status="success", data=result_, msg="Service applied successfully")
