import logging
from waitress import serve
from itertools import islice
from flask import Flask, jsonify, request
from texts_processing import TextsTokenizer
from flask_restplus import Api, Resource, fields


def shingle_func(splited_text: [], shingle: int) -> []:
    """"""
    if len(splited_text) <= shingle:
        return ["".join(splited_text)]
    else:
        shingles_list = [list(islice(splited_text, i, i + shingle)) for i in range(len(splited_text))]
        return ["".join(l) for l in shingles_list if len(l) == shingle]


logger = logging.getLogger("app_duplisearcher")
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

name_space = api.namespace('api', 'На вход поступает JSON, возвращает JSON')

input_data = name_space.model("Input JSONs",
                              {"Shingle_len": fields.Integer(required=True, help="Длина шингла"),
                               "Text_1": fields.String(required=True, help="Текст для сравнения 1"),
                               "Text_2": fields.String(required=True, help="Текст для сравнения 2")})

tokenizer = TextsTokenizer()


@name_space.route('/')
class Shingles(Resource):
    """"""

    @name_space.expect(input_data)
    def post(self):
        json_data = request.json
        tx1 = json_data["Text_1"]
        tx2 = json_data["Text_2"]
        shingles_len = json_data["Shingle_len"]

        lm_tx1 = tokenizer([tx1])[0]
        lm_tx2 = tokenizer([tx2])[0]

        t1_shingles = set(shingle_func(lm_tx1, shingles_len))
        t2_shingles = set(shingle_func(lm_tx2, shingles_len))
        intersection = t1_shingles & t2_shingles
        union = t1_shingles.union(t2_shingles)

        return jsonify({"text1_in_text2": len(intersection) / len(t1_shingles),
                        "text2_in_text1": len(intersection) / len(t2_shingles),
                        "jaccard": len(intersection) / len(union)})


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
    # app.run(host='0.0.0.0', port=8080)
