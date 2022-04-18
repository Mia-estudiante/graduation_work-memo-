import json


def refine_data(raw_file_loc: str, refine_file_loc: str):
    """
    제공 데이터는 다음과 같은 구조로 이루어져 있음

    {'profile': {'emotion': {'emotion-id': 'S05_D02_E68',
                         'situation': ['S05', 'D02'],
                         'type': 'E68'},
             'persona': {'computer': ['C01'],
                         'human': ['A02', 'G01'],
                         'persona-id': 'A02_G01_C01'},
             'persona-id': 'Pro_03719'},
     'talk': {'content': {'HS01': '아내가 드디어 출산하게 되어서 정말 신이 나.',
                          'HS02': '아 지금 정말 신이 나.',
                          'HS03': '아기가 점점 클게 벌써 기대가 되네. 내가 많이 놀아줘야지. ',
                          'SS01': '아내분이 출산을 하시는군요. 정말 축하드려요.',
                          'SS02': '잘 된 일이네요.',
                          'SS03': '좋은 아빠가 되실 거 같아요. 진심으로 축하드려요.'},
              'id': {'profile-id': 'Pro_03719', 'talk-id': 'Pro_03719_00016'}}}

    profile.emotion.type 가 감정임, 감정은 E10~E69 까지 있으므로 0~59까지 레이블링하도록 함
    """

    with open(raw_file_loc, "r") as raw_train_file:
        data = "".join(raw_train_file.readlines())
        jsons = json.loads(data)

    with open(refine_file_loc, "w") as refined_train_file:
        refined_train_file.write("Q,A,label")
        for line in jsons:
            emotion = int(line["profile"]["emotion"]["type"][1:]) - 10
            for i in range(3):
                question: str = line["talk"]["content"][f"HS0{i + 1}"]
                question = question.strip()
                response: str = line["talk"]["content"][f"SS0{i + 1}"]
                response = response.strip()

                if len(question) > 0 and len(response) > 0:

                    refined_train_file.write(f"\n{question},{response},{emotion}")


if __name__ == "__main__":

    refine_data(
        raw_file_loc='./sentimental_data/감성대화말뭉치(최종데이터)_Training.json',
        refine_file_loc='./refined_sentimental/train.csv'
    )

    refine_data(
        raw_file_loc='./sentimental_data/감성대화말뭉치(최종데이터)_Validation.json',
        refine_file_loc='./refined_sentimental/test.csv'
    )

    # with open('./refined_sentimental/train.csv', "r") as test:
    #     data = test.readlines()
    #     for line in data:
    #         d = line.split(",")
    #         if len(d) > 3:
    #             print(d)



    "python kobart_chit_chat.py --gradient_clip_val 1.0 --max_epochs 2 --default_root_dir logs --model_path kobart_from_pretrained --tokenizer_path emji_tokenizer --chat --train_file refined_sentimental/train.csv --test_file refined_sentimental/test.csv --gpus 1"
