import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data.csv')
data = data.sample(frac=1).reset_index(drop=True)

label_encoders = {}
for column in ['VISIT_AREA_NM']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data[['GENDER', 
          'TRAVEL_STYL_1', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
          'RESIDENCE_TIME_MIN', 'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']]
y = data['VISIT_AREA_NM']

model = RandomForestClassifier(n_estimators=235, max_depth=20, min_samples_split=2, min_samples_leaf=2, random_state=42)
model.fit(X, y)
inputs = None
ouputs = None
def predict_area(gender, travel_style_1, travel_style_5, travel_style_6, travel_style_7, travel_style_8):
    default_residence_time_min = data['RESIDENCE_TIME_MIN'].mean()
    default_dgstfn = data['DGSTFN'].mean()
    default_revisit_intention = data['REVISIT_INTENTION'].mean()
    default_rcmdtn_intention = data['RCMDTN_INTENTION'].mean()

    user_profile = np.array([[gender, travel_style_1, travel_style_5, travel_style_6, travel_style_7, travel_style_8,
                              default_residence_time_min, default_dgstfn, default_revisit_intention, default_rcmdtn_intention]])

    user_input_df = pd.DataFrame(user_profile, columns=['GENDER', 'TRAVEL_STYL_1', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6',
                                                        'TRAVEL_STYL_7', 'TRAVEL_STYL_8', 'RESIDENCE_TIME_MIN',
                                                        'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION'])

    user_prediction_probs = model.predict_proba(user_input_df)

    top_indices = np.argsort(-user_prediction_probs, axis=1)[:, :3].flatten()
    top_areas = label_encoders['VISIT_AREA_NM'].inverse_transform(top_indices)

    #bot_indices = np.argsort(-user_prediction_probs, axis=1)[:, -5:].flatten()
    #bot_areas = label_encoders['VISIT_AREA_NM'].inverse_transform(bot_indices)
    #for i in range( len(list(top_areas)) ):
    #  top_areas[i] = top_areas[i] + ","
        
    combined_recommendations = list(top_areas) #+ list(bot_areas)
    np.random.shuffle(combined_recommendations)
#"결과를 복사해서 구글폼 1번 문항에 붙여넣어주세요!\n" +
    formatted_output =  "다음과 같은 곳을 추천 드릴게요!\n"+"\n".join(f"{i+1}. {area}" for i, area in enumerate(combined_recommendations))
    return formatted_output

iface = gr.Interface(
    fn=predict_area,
    inputs=[
        gr.Radio(choices=[("남성", 0), ("여성", 1)], label="성별을 선택하세요"),
        gr.Slider(minimum=1, maximum=7, step=1, value=4, label="자연 선호 여부를 입력하세요 (<-자연선호 | 도시선호->)"),
        gr.Slider(minimum=1, maximum=7, step=1, value=4, label="휴양 선호 여부를 입력하세요 (<-휴양선호 | 체험선호->)"),
        gr.Slider(minimum=1, maximum=7, step=1, value=4, label="여행지 스타일을 입력하세요 (<-잘 알려지지 않은 여행지 선호 | 잘 알려진 여행지 선호->)"),
        gr.Slider(minimum=1, maximum=7, step=1, value=4, label="여행 스타일을 입력하세요 (<-계획에 따른 여행 선호 | 상황에 따른 여행 선호->)"),
        gr.Slider(minimum=1, maximum=7, step=1, value=4, label="사진 중요 여부를 입력하세요 (<-중요 | 중요하지 않음->)")
    ],
    outputs="text",
    title="여행지 추천 AI",
    description="여러분의 선호도에 맞춰 최적의 여행지를 추천해드립니다.",
)

with gr.Blocks() as demo:
  iface = gr.Interface(
    fn=predict_area,
    inputs=[
        gr.Radio(choices=[("남성", 0), ("여성", 1)], label="성별을 선택하세요"),
        gr.Slider(minimum=1, maximum=7, step=1, value=4, label='''자연 vs 도시 
                           (1에 가까울수록 자연 선호, 7에 가까울수록 도시 선호)'''),
        gr.Slider(minimum=1, maximum=7, step=1, value=4, label="휴양 vs 체험활동 \n(1에 가까울수록 휴양 선호, 7에 가까울수록 체험활동 선호)"),
        gr.Slider(minimum=1, maximum=7, step=1, value=4, label="잘 알려진 여행지 vs 잘 알려지지 않은 여행지 \n(1에 가까울수록 잘 알려진 여행지 선호, 7에 가까울수록 잘 알려지지 않은 여행지 선호)"),
        gr.Slider(minimum=1, maximum=7, step=1, value=4, label="계획 vs 상황 \n(1에 가까울수록 계획에 따른 여행 선호, 7에 가까울수록 상황에 따른 여행 선호)"),
        gr.Slider(minimum=1, maximum=7, step=1, value=4, label="사진 촬영 중요 vs 사진 촬영 중요하지 않음 \n(1에 가까울수록 사진 촬영 중요, 7에 가까울수록 사진 촬영 중요하지 않음)")
    ],
    
    outputs="text",
    title="여행지 추천 AI",
    description="여러분의 선호도에 맞춰 최적의 여행지를 추천해드립니다. ",
    allow_flagging ="never" )

# Launch Gradio interface
demo.launch(share=True)
