
def main():
    import matplotlib.pyplot as plt
    from transformers import AutoConfig
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TimeLLM
    from neuralforecast.utils import AirPassengersPanel
    from transformers import AutoConfig
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import yfinance as yf
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST
    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.utils import augment_calendar_df
    import matplotlib as mpl
    from datetime import datetime, date
    import sys
    import json
    import pandas as pd
    import requests
    print(sys.argv)
    string_value = sys.argv[1]
    # CSV 파일 읽기
    df = pd.read_csv('transcript_with_page.csv')

    grouped = df.groupby('page', as_index=False).agg({
        'text': ' '.join  
    })
    #example_df = pd.read_csv('example2.csv')
    example_df = grouped
    page_df = pd.read_csv('test1_text_by_page.csv')

    # Ollama 서버 URL
    OLLAMA_URL = 'http://localhost:11434/api/generate'

    # 결과 저장용 리스트
    results = []
    valuecount = 0
    # 예시로 example2.csv의 각 row마다 실행
    for idx, row in example_df.iterrows():
        text1 = row['text']  # example2.csv에는 'text' 컬럼이 있다고 가정

        page_num = row['page']  # page 번호
        # page_df에서 해당하는 page의 text 가져오기
        matched_rows = page_df[page_df['page'] == page_num]

        if matched_rows.empty:
            print(f"page {page_num}에 해당하는 텍스트가 없습니다.")
            continue

        text2 = matched_rows.iloc[0]['text']

        # 프롬프트 구성
        prompt = f"""
        문장1의 내용을 바탕으로 문장2를 보완해줘. 단, 문장1에 없는 내용은 추가하지 말고 문장2 안의 표현이나 맥락을 더 구체화하거나 풍부하게 해줘.

        결과는 보완된 문장2만 출력해. 시스템 프롬프트나 설명은 생략하고, 너의 사고과정도 드러내지 마.
        {sys.argv[0]}
        문장1: {text1}
        문장2: {text2}
        """

        # 요청 payload
        payload = {
            "model": "gemma3:12b",
            "prompt": prompt,
            "stream": False
        }

        # 요청 보내기
        response = requests.post(OLLAMA_URL, json=payload)

        # 응답 처리
        if response.status_code == 200:
            result = response.json()
            print(f"(index {idx}):")
            print(result['response'])
            results.append({
                'index': idx,
                'page': page_num,
                'text1': text1,
                'text2': text2,
                'gemma_response': result['response']
            })
        else:
            print(f"요청 실패 (index {idx}): {response.status_code}")
            print(response.text)
        if valuecount == 1:
            break

        valuecount += 1

    # 필요하면 결과를 CSV로 저장
    output_df = pd.DataFrame(results)
    #output_df.to_csv('gemma_opinions.csv', index=False)
    return output_df

if __name__ == "__main__":
    result_df = main()
    #result = result_df.to_dict(orient="records")
    #import json
    #print(json.dumps(result, ensure_ascii=False, default=str))

