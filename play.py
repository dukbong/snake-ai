"""
학습된 모델로 Snake 게임을 플레이하며 구경하는 스크립트.

실행:
    python play.py            # 무한 반복
    python play.py --games 5  # 5게임만
"""
import argparse
import torch
import numpy as np
from game import SnakeGameAI, BLOCK_SIZE
from model import ActorCritic

CHECKPOINT_FILE = './model/model_ppo.pth'
ACTION_MAP = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# 커리큘럼 그리드 (main.py와 동일)
CURRICULUM_GRIDS = [
    (8, 8), (12, 12), (16, 16), (20, 24), (24, 32),
]


def play():
    parser = argparse.ArgumentParser(description='Snake AI 플레이 관람')
    parser.add_argument('--games', type=int, default=0,
                        help='플레이할 게임 수 (기본: 0 = 무한)')
    parser.add_argument('--grid', type=str, default=None,
                        help='그리드 크기 (예: 8x8, 24x32). 미지정 시 체크포인트 스테이지 사용')
    args = parser.parse_args()

    checkpoint = torch.load(CHECKPOINT_FILE, weights_only=True)

    # 커리큘럼 스테이지 복원
    curriculum_stage = checkpoint.get('curriculum_stage', len(CURRICULUM_GRIDS) - 1)
    frame_stack_n = checkpoint.get('frame_stack_n', 1)

    if args.grid:
        parts = args.grid.lower().split('x')
        grid_rows, grid_cols = int(parts[0]), int(parts[1])
    else:
        grid_rows, grid_cols = CURRICULUM_GRIDS[curriculum_stage]

    in_channels = 3 * frame_stack_n + 4 if frame_stack_n > 1 else 7

    model = ActorCritic(in_channels=in_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    record = checkpoint.get('record', 0)
    n_games = checkpoint.get('n_games', 0)
    grid_str = f"{grid_rows}×{grid_cols}"
    print(f"모델 로드 완료 — 학습 게임: {n_games}회, 최고기록: {record}")
    print(f"그리드: {grid_str}, Stage: {curriculum_stage}, Frame Stack: {frame_stack_n}")
    print("창을 닫거나 Ctrl+C로 종료")

    env = SnakeGameAI(
        w=grid_cols * BLOCK_SIZE,
        h=grid_rows * BLOCK_SIZE,
        render=True,
    )
    game_count = 0
    scores = []

    try:
        while True:
            env.reset()
            while True:
                state = env.get_grid_state()
                state_t = torch.from_numpy(state).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model(state_t)
                    action = logits.argmax(dim=-1)
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
