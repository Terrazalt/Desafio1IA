"""
Planilla para el desarollo del desafio 1:  Freno de emergencia.

"""

import argparse
import cv2
import numpy as np
import pyglet

from pyglet.window import key
from pyglet.gl import *
from gym_duckietown.envs import DuckietownEnv


FILTRO_AMARILLO_LOWER = np.array([20, 100, 120])  # Ajusta el valor mínimo del tono (Hue)
FILTRO_AMARILLO_UPPER = np.array([35, 255, 255])  #

class Desafio1:
    
    def __init__(self, map_name):
        self.env = DuckietownEnv(
            seed=1,
            map_name=map_name,
            draw_curve=False,
            draw_bbox=False,
            domain_rand=False,
            frame_skip=1,
            distortion=False,
        )

        # Esta variable nos ayudara a definir la acción que se ejecutará en el siguiente paso (loop)
        self.last_obs = self.env.reset()
        self.env.render()

        # Registrar el handler
        self.key_handler = key.KeyStateHandler()
        self.env.unwrapped.window.push_handlers(self.key_handler)


    def get_mask(self,obs: np.array) -> np.array:
        FILTRO_VERDE_LOWER = np.array([31, 0, 0])  # Ajusta estos valores al color del pasto
        FILTRO_VERDE_UPPER = np.array([179, 255, 255])  # Ajusta estos valores según sea necesario

        hsv_image = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
        mask_verde = cv2.inRange(hsv_image, FILTRO_VERDE_LOWER, FILTRO_VERDE_UPPER)
        mask = cv2.inRange(hsv_image, FILTRO_AMARILLO_LOWER, FILTRO_AMARILLO_UPPER)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_verde))
        return mask
    
    
     
    def morph_ops(self, mask: np.array) -> np.array:
        kernel = np.ones((5, 5), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations=1)  # Erosión
        mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=3)  # Dilatación
        return mask_dilated



    def emergency_brake(self, obs: np.array) -> bool:
        """
        Método que implementa el freno de emergencia. Dada la observación del agente, se debe
        determinar si se activa el freno de emergencia o no.

        """        
        mask = self.get_mask(obs)
        mask_morph = self.morph_ops(mask)

        contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_total = obs.shape[0]*obs.shape[1]
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter>500:
                cv2.rectangle(obs, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(obs, "Patito", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            obs_rgb = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
            cv2.imshow('Patitos',obs_rgb)
            if area/area_total >= 0.09: 
                print(perimeter)
                print(area)   
                return True

        return False


    def update(self, dt):
        """
        Este método se encarga de ejecutar un loop de simulación en el ambiente Duckietown.
        En cada paso se ejecuta una acción que se define en este método y se obtienen las 
        observaciones del agente en el ambiente.

        Este método debe usar la última observación obtenida por el agente (la variable
        self.last_obs) y realizar una acción en base a esta. Luego debe guardar la observación
        actual para la siguiente iteración.

        """

        if self.last_obs is None:
            self.last_obs = self.env.reset()

        action = np.array([0.0, 0.0])
        
        # Tele - operación: Control manual del agente dado por las teclas
        if self.key_handler[key.UP]:
            action[0] += 0.44

        if self.key_handler[key.DOWN]:
            action[0] -= 0.44

        if self.key_handler[key.LEFT]:
            action[1] += 1

        if self.key_handler[key.RIGHT]:
            action[1] -= 1

        if self.key_handler[key.SPACE]:
            action = np.array([0, 0])

        # Speed boost
        if self.key_handler[key.LSHIFT]:
            action *= 1.5

        # Se ejecuta el freno de emergencia
        if self.emergency_brake(self.last_obs):
            if self.key_handler[key.UP]:
                action[0] = 0
            
                

        # Aquí se obtienen las observaciones y se fija la acción
        # obs consiste en un imagen de 640 x 480 x 3
        self.last_obs, _, done, _ = self.env.step(action)
        self.env.render()

        if done:
            self.last_obs = self.env.reset()

    def run(self):
        """
        Arranca la simulación del ambiente Duckietown.
        """

        # Fijar la frecuencia de la simulación. Se ejecutara el método update() cada 1/fps segundos
        pyglet.clock.schedule_interval(self.update, 1.0 / self.env.unwrapped.frame_rate)
        pyglet.app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-name",
        default="loop_obstacles",
        help=(
            "Nombre del mapa donde correr la simulación. El mapa debe "
            "estar en la carpeta de mapas (gym_duckietown/maps.)."
        ),
    )

    args = parser.parse_args()
    Desafio1(map_name=args.map_name).run()
