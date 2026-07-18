import lmstudio as lms

model = lms.llm("llava-llama-3-8b-v1_1")

image_handle = lms.prepare_image(r"C:\Users\user\Downloads\AAS_CV\Indonesian License Plate Recognition Dataset\images\test\test007_1.jpg")

chat = lms.Chat(
    "You are an OCR system. Your ONLY task is to read the license plate number in the image. "
    "Do not describe the image. Do not mention location, background, or context. "
    "Do not explain anything. Output ONLY the license plate number characters, nothing else."
)
chat.add_user_message(
    "What is the license plate number shown in this image? Respond only with the plate number.",
    images=[image_handle]
)

prediction = model.respond(chat, config={"temperature": 0.0, "maxTokens": 15})
print(prediction)