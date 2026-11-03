import cv2
import mediapipe as mp
import pygame
import sys

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # pinky
]

pygame.init()
WIDTH, HEIGHT = 1280, 720
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Hand Sensing Sandbox (Multi-Hand)")
clock = pygame.time.Clock()

# wsl / linux camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font_main = pygame.font.SysFont("Menlo", 22)
font_large = pygame.font.SysFont("Menlo", 36, bold=True)

while True:
    dt = clock.tick(30)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    ret, frame = cap.read()
    window.fill((20, 24, 33))

    if ret:
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = hand_landmarks.landmark

                index_tip = landmarks[8]

                # translate & render
                pts = [(int(lm.x * WIDTH), int(lm.y * HEIGHT)) for lm in landmarks]
                
                # static colors for the skeleton
                bone_color = (0, 255, 180) 
                for p1, p2 in HAND_CONNECTIONS:
                    pygame.draw.line(window, bone_color, pts[p1], pts[p2], 4)

                for i, lm in enumerate(landmarks):
                    px, py = pts[i]
                    node_radius = max(4, int(14 - (lm.z * 60)))
                    node_color = (0, 255, 150) 

                    pygame.draw.circle(window, node_color, (px, py), node_radius)
                    pygame.draw.circle(window, (255, 255, 255), (px, py), 2)

                y_offset = hand_idx * 200

                state_str = f"HAND {hand_idx + 1}: DETECTED"
                state_color = (0, 255, 150)
                
                state_text = font_large.render(state_str, True, state_color)
                window.blit(state_text, (30, 30 + y_offset))

                hud_lines = [
                    f"Index Tip (#8) -> X: {index_tip.x:.3f} | Y: {index_tip.y:.3f} | Z: {index_tip.z:.3f}"
                ]

                for idx, line in enumerate(hud_lines):
                    txt_surface = font_main.render(line, True, (220, 220, 230))
                    window.blit(txt_surface, (30, 80 + y_offset + (idx * 28)))

    else:
        no_hand_txt = font_large.render("NO HAND DETECTED - Hold hands up to camera", True, (255, 100, 100))
        window.blit(no_hand_txt, (WIDTH // 2 - 400, HEIGHT // 2))

    pygame.display.update()