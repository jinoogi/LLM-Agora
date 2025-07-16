# 필요한 라이브러리를 불러옵니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 사용자 설정 ---
# 엑셀에서 변환된 CSV 파일 이름을 정확하게 입력해주세요.
file_name = 'score.csv'
# -----------------

try:
    # CSV 파일을 읽어옵니다.
    df_raw = pd.read_csv(file_name)

    # 'condition' 열을 만듭니다. (그래프의 범례로 사용)
    df_raw['condition'] = df_raw['agent_type'] + '-' + df_raw['cat_attack'].astype(str)
    df_raw['condition'] = df_raw['condition'].replace({'True': 'cat_attack', 'False': 'no_cat_attack'}, regex=True)

    # 여러 개의 그래프를 생성합니다.
    # col='temp' : 온도별로 그래프를 나눕니다.
    # kind='line' : 꺾은선 그래프를 그립니다.
    # errorbar='sd' : 표준편차를 오차 범위(음영)로 표시합니다.
    g = sns.relplot(
        data=df_raw,
        x='round',
        y='score',
        hue='condition',
        style='condition',
        col='temp',
        kind='line',
        errorbar='sd',
        markers=True,
        height=6,
        aspect=0.8
    )

    # 그래프의 제목과 축 레이블을 설정합니다.
    g.fig.suptitle('Mean Performance with Standard Deviation for Each Temperature', y=1.03)
    g.set_axis_labels('Round', 'Mean Score')
    g.set_titles('Temp = {col_name}')
    g.legend.set_title('Condition')

    # 완성된 그래프를 이미지 파일로 저장합니다.
    output_filename = 'all_temps_lineplot_with_std.png'
    plt.savefig(output_filename, bbox_inches='tight')

    print(f"그래프가 성공적으로 '{output_filename}' 파일로 저장되었습니다.")

except FileNotFoundError:
    print(f"오류: '{file_name}' 파일을 찾을 수 없습니다.")
    print("스크립트와 데이터 파일이 같은 폴더에 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")