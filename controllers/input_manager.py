import carla
import pygame


class InputManager:
    def __init__(self, window_title: str = "CARLA Controls"):
        pygame.init()
        pygame.display.set_mode((320, 120))
        pygame.display.set_caption(window_title)
        self.last_exit_reason = None
        self.running = True

    def poll(self):
        toggle_auto = False

        if self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.last_exit_reason = "Pygame control window was closed"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        self.last_exit_reason = "Escape key was pressed"
                    elif event.key == pygame.K_e:
                        toggle_auto = True

        if not self.running:
            return False, False, carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                reverse=False,
                hand_brake=False,
                manual_gear_shift=False,
            )

        keys = pygame.key.get_pressed()

        throttle = 0.0
        steer = 0.0
        brake = 0.0
        reverse = False

        if keys[pygame.K_w]:
            throttle = 0.7
        if keys[pygame.K_s]:
            reverse = True
            throttle = 0.5
        if keys[pygame.K_a]:
            steer = -0.5
        if keys[pygame.K_d]:
            steer = 0.5
        if keys[pygame.K_SPACE]:
            brake = 1.0
            throttle = 0.0

        manual_control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            reverse=reverse,
            hand_brake=False,
        )

        return self.running, toggle_auto, manual_control

    def is_key_pressed(self, key_char: str) -> bool:
        keys = pygame.key.get_pressed()
        
        # Convert the character to its pygame key constant
        if len(key_char) == 1:
            if key_char == ' ':
                key_constant = pygame.K_SPACE
            else:
                # Convert the character to its pygame key constant
                key_constant = pygame.key.key_code(key_char)
        else:
            # Handle special cases or invalid input
            return False
        
        return keys[key_constant]

    def shutdown(self):
        pygame.quit()