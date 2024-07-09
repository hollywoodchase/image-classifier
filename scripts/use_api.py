import requests

def classify_image(image_path):
    url = 'http://localhost:5000/predict'
    with open(image_path, 'rb') as img:
        files = {'file': img}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        print(f"Class: {data['class']}")
        print(f"Confidence: {data['confidence']:.2f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 use_api.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    classify_image(image_path)
