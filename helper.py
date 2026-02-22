import matplotlib
matplotlib.use('Agg')  # 헤드리스 환경용 (화면 없이도 동작)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

plt.ion()

# 한글 폰트 설정
def _set_korean_font():
    korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 'AppleGothic']
    for fname in korean_fonts:
        try:
            fm.findfont(fm.FontProperties(family=fname), fallback_to_default=False)
            plt.rcParams['font.family'] = fname
            return
        except Exception:
            continue
    # 한글 폰트가 없으면 영문으로 대체
    plt.rcParams['font.family'] = 'DejaVu Sans'

_set_korean_font()
plt.rcParams['axes.unicode_minus'] = False

scores_history = []
mean_scores_history = []


def plot(scores, mean_scores):
    scores_history.clear()
    scores_history.extend(scores)
    mean_scores_history.clear()
    mean_scores_history.extend(mean_scores)

    plt.clf()
    plt.title('Snake AI Training Progress')
    plt.xlabel('Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', alpha=0.7)
    plt.plot(mean_scores, label='Mean Score', linewidth=2)
    plt.ylim(ymin=0)
    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    plt.legend()
    plt.tight_layout()

    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/training_progress.png', dpi=80)
    plt.pause(0.001)
