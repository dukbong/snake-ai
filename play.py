"""
학습된 모델로 Snake 게임을 플레이하며 구경하는 스크립트.

실행:
    python play.py            # 무한 반복
    python play.py --games 5  # 5게임만
"""
import argparse
import torch
import numpy as np
from game import SnakeGameAI
from model import ActorCritic

CHECKPOINT_FILE = './model/model_ppo.pth'
ACTION_MAP = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def play():
    parser = argparse.ArgumentParser(description='Snake AI 플레이 관람')
    parser.add_argument('--games', type=int, default=0,
                        help='플레이할 게임 수 (기본: 0 = 무한)')
    args = parser.parse_args()

    model = ActorCritic()
    checkpoint = torch.load(CHECKPOINT_FILE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    record = checkpoint.get('record', 0)
    n_games = checkpoint.get('n_games', 0)
    print(f"모델 로드 완료 — 학습 게임: {n_games}회, 최고기록: {record}")
    print("창을 닫거나 Ctrl+C로 종료")

    env = SnakeGameAI(render=True)
    game_count = 0
    scores = []

    try:
        while True:
            env.reset()
            while True:
                state = env.get_grid_state()
                state_t = torch.from_numpy(state).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = model.get_action_and_value(state_t)
                _, done, _, score = env.play_step(ACTION_MAP[action.item()])
                if done:
                    break

            game_count += 1
            scores.append(score)
            print(f"게임 {game_count:3d} | 점수: {score:3d} | 평균: {np.mean(scores):.1f} | 최고: {max(scores)}")

            if args.games and game_count >= args.games:
                break

    except KeyboardInterrupt:
        pass

    if scores:
        print(f"\n총 {game_count}게임 | 평균: {np.mean(scores):.1f} | 최고: {max(scores)}")


if __name__ == '__main__':
    play()
