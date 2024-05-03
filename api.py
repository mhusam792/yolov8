from PIL import Image
import requests
from io import BytesIO
from fastapi import HTTPException
# from fastapi.responses import JSONResponse
# import os
from yolov8.image_predictor import ImagePredictor


model_path = '/content/yolov8/last.pt'
image_predictor = ImagePredictor(model_full_path=model_path)

def read_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for errors in the HTTP response

        image = Image.open(BytesIO(response.content))
        return image

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image from URL: {str(e)}")
    

# Categories
def categorize_items(item_dict):
    type_list = []

    if len(item_dict) == 0:
        return type_list

    categories = dict(
        bags = ["Backpack", "Briefcase", "Handbag", "Suitcase"],
        electronics = ["Computer keyboard", "Computer monitor", "Computer mouse", "Tablet computer", "Calculator", "Camera", "Tripod", "Laptop", "Mobile phone", "Headphones", "Flashlight"],
        animals = ["Cat", "Dog"],
        accessories = ["Fashion accessory", "Earrings", "Necklace"],
        hats = ["Hat", "Fedora", "helmet", "Bicycle helmet", "Sun hat", "Swim cap"],
        glasses = ["Glasses", "Sunglasses", "Binoculars"],
        toys = ["Doll", "Football", "Ball", "Volleyball (Ball)"],
        personal_purposes = ["Watch", "wallet", "Bottle", "Mug", "Pencil case", "Book", "Umbrella", "Envelope"],
        clothes = ["Tie", "Belt"]
    )

    type_list = []

    for item, _ in item_dict.items():
        found = False
        for category, items in categories.items():
            if item in items:
                type_list.append(category)
                found = True
                break
        if not found:
            type_list.append("others")

    return type_list


# def yolo_api(FastApi, model_full_path):
#     @app.get("/predict", tags=["Model"])
#     async def predict_image(file: str):
def yolo_model(model_full_path, image: str):
    # try:
    prediction_result = image_predictor.model_result(
        model_path=model_full_path,
        img_path=read_image_from_url(image)
    )
    
    cls = image_predictor.return_cls(
        model_result=prediction_result[0],
        trained_model=prediction_result[1]
    )
    dict_cls = image_predictor.count_cls(cls)

    if len(dict_cls) == 0:
        print("Sorry, We can't figure out what type of item it is?! \n Please choose from this list the type of item in the image.")
        founded_objects = dict_cls
    else:
        founded_objects = dict_cls

    # if not os.path.exists(self.source_folder_path):
    #     os.makedirs(self.source_folder_path)

    # self.image_predictor.copy_images_and_remove_folder(
    #     self.source_folder_path, self.destination_folder_path
    # )

    # path_all_images = 'folders/runs/detect/prediction/' ###################
    # image_full_path = os.path.join(path_all_images, file)
    # file_name_without_extension = os.path.splitext(os.path.basename(image_full_path))[0]



    type_list = categorize_items(founded_objects)
    
    data_from_image = {"objects": founded_objects, "type": type_list}
    
    return data_from_image

    #     return JSONResponse(content=data_from_image)

    # except Exception as e:
    #     return JSONResponse(content={"error": str(e)}, status_code=500)
