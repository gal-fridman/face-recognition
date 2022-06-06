import numpy as np
from flask import Flask, request
import json
from face_recognition_funcs import find_matching_image, images_to_encodings
import pandas as pd

app = Flask(
    __name__,
)


def to_native_types(method_outputs: dict):
    def do_nothing(obj):
        return obj

    def time_series_to_json(series: pd.Series):
        df = series.to_frame()
        return time_data_frame_to_json(df)

    def time_data_frame_to_json(df: pd.DataFrame):
        df.index = df.index.rename('data').astype('str')
        df = df.reset_index()
        return {
            'data': df.values.tolist(),
            'data_map': df.columns.tolist()
        }

    type_handler_mapper = {
        pd.DataFrame: time_data_frame_to_json,
        pd.Series: time_series_to_json,
        np.ndarray: np.ndarray.tolist,
        dict: to_native_types
    }
    json_out = {}
    for k, v in method_outputs.items():
        json_out[k] = type_handler_mapper.get(type(v), do_nothing)(v)
    return json_out


@app.route("/health", methods=['GET'])
def health():
    return 'OK'


@app.route("/find_matching_image", methods=['POST'])
def run_find_matching_image():
    data_payload = json.loads(request.data)
    df = pd.DataFrame.from_dict(data_payload['data'], orient='index')
    result = find_matching_image(data_payload['image_url'], df.encodings.to_dict())
    return to_native_types(result)


@app.route("/images_to_encodings", methods=['POST'])
def run_images_to_encodings():
    data_payload = json.loads(request.data)
    result = images_to_encodings(**data_payload)
    return to_native_types(result)

#
if __name__ == '__main__':
    app.run(port=8080, debug=True)
