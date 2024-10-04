#!/usr/bin/env python3
import subprocess as spr
import os

spr.call(["nohup python3 driver_ball_OH_1.py 0 > trace0.out"], shell=True)

spr.call(["nohup python3 driver_ball_OH_2.py 3 > trace1.out"], shell=True)

#spr.call(["nohup python3 driver_ball_OH_3.py 6 > trace2.out"], shell=True)

