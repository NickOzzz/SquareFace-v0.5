from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager, SlideTransition
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image, AsyncImage
from kivy.uix.filechooser import FileChooserListView, FileChooserIconView
from kivy.uix.popup import Popup
import cv2
import os
import pygame
import ffmpeg
from moviepy.audio.io.AudioFileClip import AudioFileClip
from kivy.config import Config
import time
import numpy
import sys
from kivy.clock import Clock
from keras.models import model_from_json
import keras.preprocessing.image as ima


switcher = ""
folder_path = ""
classifier = ""
txt1 = ""
txt2 = ""
app = ""
app_assets = ""
screen_to = "choose"


class welcome(Screen):
    def __init__(self, **kwargs):
        super(welcome, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        label1 = Label(text="WELCOME TO SQUARE FACE", pos_hint={"x": -0.003, "y": 0.20}, color=(0.309, 0.933, 0.078, 4))
        label2 = Label(text="choose what you want to detect or recognize", pos_hint={"x": -0.004, "y": 0.123},
                       color=(0.309, 0.933, 0.078, 4))
        but1 = Button(text="FACE", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.04, "y": 0.38},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.20, 0.15))
        but1.bind(on_press=self.switchtoface)
        but2 = Button(text="BODY", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.28, "y": 0.38},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.20, 0.15))
        but2.bind(on_press=self.switchtobody)
        but3 = Button(text="EYE", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.52, "y": 0.38},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.20, 0.15))
        but3.bind(on_press=self.switchtoeye)
        but4 = Button(text="SMILE", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.76, "y": 0.38},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.20, 0.15))
        but4.bind(on_press=self.switchtosmile)
        but5 = Button(text="LICENSE PLATE AND IT'S NUMBER", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.358, "y": 0.10},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="13sp")
        but5.bind(on_press=self.switch)
        but6 = Button(text="FACE AND PERSON'S IDENTITY", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.665, "y": 0.10},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="13sp")
        but6.bind(on_press=self.switch)
        but7 = Button(text="EMOTION", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.05, "y": 0.10},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="13sp")
        but7.bind(on_press=self.switchtoemotion)
        img = Image(source=app_assets + "assets/logo.png", size_hint=(0.15, .3), pos_hint={"x": 0.423, "y": 0.71})

        layout.add_widget(label1)
        layout.add_widget(label2)
        layout.add_widget(but1)
        layout.add_widget(but2)
        layout.add_widget(but3)
        layout.add_widget(but4)
        layout.add_widget(but5)
        layout.add_widget(but6)
        layout.add_widget(but7)
        layout.add_widget(img)
        self.add_widget(layout)

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "premium"

    def switchtoface(self, *args):
        global switcher
        switcher = "face"
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "choose"

    def switchtobody(self, *args):
        global switcher
        switcher = "body"
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "choose"

    def switchtoeye(self, *args):
        global switcher
        switcher = "eye"
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "choose"

    def switchtosmile(self, *args):
        global switcher
        switcher = "smile"
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "choose"

    def switchtoemotion(self, *args):
        global switcher
        switcher = "emotion"
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "choose"


class choose(Screen):
    def __init__(self, **kwargs):
        super(choose, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        label1 = Label(text="CHOOSE TYPE OF FILE", pos_hint={"x": -0.003, "y": 0.27}, color=(0.309, 0.933, 0.078, 4))
        but1 = Button(text="VIDEO", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.20, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20))
        but1.bind(on_press=self.switch4)
        but2 = Button(text="VIDEO FROM YOUR CAMERA", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12sp")
        but2.bind(on_press=self.switch5)
        but3 = Button(text="IMAGE", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.20, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20))
        but3.bind(on_press=self.switch2)
        but4 = Button(text="IMAGE FROM YOUR CAMERA", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12sp")
        but4.bind(on_press=self.switch3)
        but5 = Button(text="< BACK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.035, "y": 0.85},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.2, 0.1))
        but5.bind(on_press=self.switch)

        layout.add_widget(label1)
        layout.add_widget(but1)
        layout.add_widget(but2)
        layout.add_widget(but3)
        layout.add_widget(but4)
        layout.add_widget(but5)
        self.add_widget(layout)

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "welcome"

    def switch2(self, *args):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "pathim"

    def switch3(self, *args):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "pathimcam"

    def switch4(self, *args):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "pathvid"

    def switch5(self, *args):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "pathvidcam"


class pathim(Screen):
    def __init__(self, **kwargs):
        super(pathim, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        self.popup_layout = FloatLayout(size=(175, 300))

        label4 = Label(text="IMAGE",
                       pos_hint={"x": 0.194, "y": 0.83},
                       color=(0.309, 0.933, 0.078, 4), size_hint=(0.62, .07))
        label3 = Label(text="WRITE FULL PATH TO AN IMAGE (i.e. /Users/joe/img.png) OR PRESS FILE ICON TO CHOOSE FILE", pos_hint={"x": 0.01, "y": 0.24},
                       color=(0.309, 0.933, 0.078, 4))
        self.txt1 = TextInput(hint_text='PATH TO AN IMAGE',
                              multiline=False,
                              size_hint=(0.55, .07),
                              pos_hint={'x': 0.2, 'y': 0.57},
                              background_color=(0.309, 0.933, 0.078, 4))
        but1 = Button(text="SAVE FULL IMAGE >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.33},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        but1.bind(on_press=self.scan)
        but3 = Button(text="SHOW RESULT", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.20, "y": 0.33},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        but3.bind(on_press=self.result)
        but4 = Button(text="SHOW ONLY DETECTED AREAS", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.08},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        but4.bind(on_press=self.result_area)
        but5 = Button(text="SAVE ONLY DETECTED AREAS >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.08},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        but5.bind(on_press=self.scan_area)
        but2 = Button(text="< BACK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.035, "y": 0.85},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.2, 0.1))
        but2.bind(on_press=self.switch)
        file_finder = Button(text="", background_normal=app_assets + "assets/file.png", background_down=app_assets + "assets/file.png", pos_hint={"x": 0.75, "y": 0.56}, size_hint=(0.075, 0.095))
        file_finder.bind(on_press=self.pops)
        self.file_chooser = FileChooserIconView(dirselect=True)
        but_close_popup = Button(text="CANCEL", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.005, "y": 1.015},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.15, 0.05))
        but_ok_popup = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4),
                                 pos_hint={"x": 0.84, "y": 1.015},
                                 color=(0.141, 0.054, 0.078, 4), size_hint=(0.15, 0.05))
        but_close_popup.bind(on_press=self.close)
        but_ok_popup.bind(on_press=self.choose_file)
        layout.add_widget(self.txt1)
        layout.add_widget(label3)
        layout.add_widget(label4)
        layout.add_widget(but1)
        layout.add_widget(but2)
        layout.add_widget(but3)
        layout.add_widget(but4)
        layout.add_widget(but5)
        layout.add_widget(file_finder)
        self.popup_layout.add_widget(self.file_chooser)
        self.popup_layout.add_widget(but_close_popup)
        self.popup_layout.add_widget(but_ok_popup)
        self.add_widget(layout)
        self.model = ""
        self.popup = Popup(title="select your file", content=self.popup_layout)
        # self.event = ""

    # def on_enter(self, *args):
    #    global screen_to
    #    try:
    #        self.event = Clock.schedule_interval(self.check, 0.5)
    #        os.listdir(app + "TO_PROCESS")
    #        os.listdir(app + "PROCESSED")
    #    except Exception as e:
    #        screen_to = "pathim"
    #        self.manager.transition = SlideTransition(direction="down")
    #        self.manager.current = "no_folder_error"

    # def on_leave(self, *args):
    #    self.event.cancel()

    def on_enter(self, *args):
        global app
        global app_assets
        if getattr(sys, 'frozen', False):
            app = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))) + "/"
            app_assets = str(os.path.dirname(sys.executable)) + "/"
        elif __file__:
            app = str(os.path.dirname(__file__)) + "/"
            app_assets = str(os.path.dirname(__file__)) + "/"
        self.model = model_from_json(open(app_assets + "assets/neuralnet.json", "r").read())
        self.model.load_weights(app_assets + "assets/weights.h5")

    def pops(self, *args):
        self.popup.open()

    def close(self, *args):
        self.popup.dismiss()

    def choose_file(self, *args):
        try:
            self.txt1.text = self.file_chooser.selection[0]
            self.popup.dismiss()
        except Exception as e:
            self.txt1.text = ""
            self.popup.dismiss()

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "choose"

    def scan(self, *args):
        app = self.txt1.text
        try:
            image = cv2.imread(app)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if switcher != "emotion":
                global classifier
                if switcher == "face":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                elif switcher == "body":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                elif switcher == "eye":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                elif switcher == "smile":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")

                detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                if detection == ():
                    self.manager.transition = SlideTransition(direction="down")
                    self.manager.current = "nodetections"
                else:
                    for (x, y, z, w) in detection:
                        cv2.rectangle(image, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                    cv2.imwrite(os.path.dirname(app) + "/ProcessedImage=).png", image)
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "finalim"
            elif switcher == "emotion":
                face_recogn = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                detection = face_recogn.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
                for (x, y, z, w) in detection:
                    cv2.rectangle(image, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                    cut_gray = gray[y: y + z, x: x + w]
                    resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                    array_gray_im = ima.img_to_array(resized_cut_gray)
                    array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                    array_gray_im_expanded /= 255
                    prediction = self.model.predict(array_gray_im_expanded)
                    final_prediction = numpy.argmax(prediction[0])
                    list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
                    result = list_of_emotions[final_prediction]
                    if z >= 210 and w >= 210:
                        cv2.putText(image, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                    (0, 0, 255), 2)
                    else:
                        cv2.putText(image, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (0, 0, 255), 1)
                cv2.imwrite(os.path.dirname(app) + "/ProcessedImage=).png", image)
                self.manager.transition = SlideTransition(direction="left")
                self.manager.current = "finalim"
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "error"

    def scan_area(self, *args):
        app = self.txt1.text
        try:
            image = cv2.imread(app)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if switcher != "emotion":
                global classifier
                if switcher == "face":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                elif switcher == "body":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                elif switcher == "eye":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                elif switcher == "smile":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                picker = 0
                if detection == ():
                    self.manager.transition = SlideTransition(direction="down")
                    self.manager.current = "nodetections"
                else:
                    for (x, y, z, w) in detection:
                        picker += 1
                        image_fin = image[y: y + z, x: x + w]
                        image_fin = cv2.resize(image_fin, (520, 400))
                        cv2.imwrite(os.path.dirname(app) + "/ProcessedImage=)" + str(picker) + ".png", image_fin)
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "finalim"
            elif switcher == "emotion":
                face_recogn = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                detection = face_recogn.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
                if detection == ():
                    self.manager.transition = SlideTransition(direction="down")
                    self.manager.current = "nodetections"
                else:
                    picker = 0
                    for (x, y, z, w) in detection:
                        picker += 1
                        image_f = image[y: y + z, x: x + w]
                        image_final = cv2.resize(image_f, (520, 400))
                        cut_gray = gray[y: y + z, x: x + w]
                        resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                        array_gray_im = ima.img_to_array(resized_cut_gray)
                        array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                        array_gray_im_expanded /= 255
                        prediction = self.model.predict(array_gray_im_expanded)
                        final_prediction = numpy.argmax(prediction[0])
                        list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
                        result = list_of_emotions[final_prediction]
                        cv2.putText(image_final, result, (10, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.4,
                                    (0, 0, 255), 2)
                        cv2.imwrite(os.path.dirname(app) + "/ProcessedImage=)" + str(picker) + ".png", image_final)
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "finalim"
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "error"

    def result(self, *args):
        app = self.txt1.text
        try:
            image = cv2.imread(app)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if switcher != "emotion":
                global classifier
                if switcher == "face":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                elif switcher == "body":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                elif switcher == "eye":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                elif switcher == "smile":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                t = True
                counter = 0
                if detection == ():
                    self.manager.transition = SlideTransition(direction="down")
                    self.manager.current = "nodetections"
                else:
                    for (x, y, z, w) in detection:
                        counter += 1
                        cv2.rectangle(image, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                        if counter == 1:
                            cv2.putText(image, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35,
                                        (20, 226, 20), 1)
                    cv2.imshow("PREVIEW", image)
                    cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.waitKey(0)
                    while t:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    t = False
                                    cv2.destroyAllWindows()
            elif switcher == "emotion":
                face_recogn = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                detection = face_recogn.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
                t = True
                counter = 0
                if detection == ():
                    self.manager.transition = SlideTransition(direction="down")
                    self.manager.current = "nodetections"
                else:
                    for (x, y, z, w) in detection:
                        counter += 1
                        cv2.rectangle(image, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                        if counter == 1:
                            cv2.putText(image, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35,
                                        (20, 226, 20), 1)
                        cut_gray = gray[y: y + z, x: x + w]
                        resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                        array_gray_im = ima.img_to_array(resized_cut_gray)
                        array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                        array_gray_im_expanded /= 255
                        prediction = self.model.predict(array_gray_im_expanded)
                        final_prediction = numpy.argmax(prediction[0])
                        list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
                        result = list_of_emotions[final_prediction]
                        if z >= 210 and w >= 210:
                            cv2.putText(image, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                        (0, 0, 255), 2)
                        else:
                            cv2.putText(image, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        (0, 0, 255), 1)
                    cv2.imshow("PREVIEW", image)
                    cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.waitKey(0)
                    while t:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    t = False
                                    cv2.destroyAllWindows()
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "error"

    def result_area(self, *args):
        app = self.txt1.text
        try:
            image = cv2.imread(app)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if switcher != "emotion":
                global classifier
                if switcher == "face":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                elif switcher == "body":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                elif switcher == "eye":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                elif switcher == "smile":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                t = True
                counter = 0
                tracker = 0
                counter_of_images = []
                for element in detection:
                    counter += 1
                    counter_of_images.append(counter)
                if detection == ():
                    self.manager.transition = SlideTransition(direction="down")
                    self.manager.current = "nodetections"
                else:
                    for (x, y, z, w) in detection:
                        final_val = counter_of_images[-1]
                        value = counter_of_images[tracker]
                        tracker += 1
                        image_fin = image[y: y + z, x: x + w]
                        image_fin = cv2.resize(image_fin, (520, 400))
                        if value == final_val:
                            cv2.putText(image_fin, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 11),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.30,
                                        (20, 226, 20), 1)
                        else:
                            cv2.putText(image_fin, "PRESS SPACE TO GET TO THE NEXT IMAGE", (10, 11),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.30,
                                        (20, 226, 20), 1)
                        cv2.imshow("PREVIEW", image_fin)
                        cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.waitKey(0)
                    while t:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    t = False
                                    cv2.destroyWindow("PREVIEW")
            elif switcher == "emotion":
                face_recogn = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                detection = face_recogn.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
                if detection == ():
                    self.manager.transition = SlideTransition(direction="down")
                    self.manager.current = "nodetections"
                else:
                    t = True
                    counter = 0
                    tracker = 0
                    counter_of_images = []
                    for element in detection:
                        counter += 1
                        counter_of_images.append(counter)
                    picker = 0
                    for (x, y, z, w) in detection:
                        picker += 1
                        image_f = image[y: y + z, x: x + w]
                        image_final = cv2.resize(image_f, (520, 400))
                        cut_gray = gray[y: y + z, x: x + w]
                        resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                        array_gray_im = ima.img_to_array(resized_cut_gray)
                        array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                        array_gray_im_expanded /= 255
                        prediction = self.model.predict(array_gray_im_expanded)
                        final_prediction = numpy.argmax(prediction[0])
                        list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
                        result = list_of_emotions[final_prediction]
                        cv2.putText(image_final, result, (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.4,
                                    (0, 0, 255), 2)
                        final_val = counter_of_images[-1]
                        value = counter_of_images[tracker]
                        tracker += 1
                        if value == final_val:
                            cv2.putText(image_final, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 11),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.30,
                                        (20, 226, 20), 1)
                        else:
                            cv2.putText(image_final, "PRESS SPACE TO GET TO THE NEXT IMAGE", (10, 11),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.30,
                                        (20, 226, 20), 1)
                        cv2.imshow("PREVIEW", image_final)
                        cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.waitKey(0)
                    while t:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    t = False
                                    cv2.destroyWindow("PREVIEW")
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "error"


class pathimcam(Screen):
    def __init__(self, **kwargs):
        super(pathimcam, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        label1 = Label(text="IMAGE FROM YOUR CAMERA", pos_hint={"x": 0.01, "y": 0.25},
                       color=(0.309, 0.933, 0.078, 4))
        but1 = Button(text="TAKE AND SAVE FULL IMAGE >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        but1.bind(on_press=self.scan)
        but3 = Button(text="TAKE AND SHOW RESULT", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        but3.bind(on_press=self.result)
        but4 = Button(text="TAKE AND SHOW DETECTED AREAS", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        but4.bind(on_press=self.result_area)
        but5 = Button(text="TAKE AND SAVE DETECTED AREAS >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        but5.bind(on_press=self.scan_area)
        but2 = Button(text="< BACK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.035, "y": 0.85},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.2, 0.1))
        but2.bind(on_press=self.switch)

        layout.add_widget(label1)
        layout.add_widget(but1)
        layout.add_widget(but2)
        layout.add_widget(but3)
        layout.add_widget(but4)
        layout.add_widget(but5)
        self.add_widget(layout)
        self.model = ""
        # self.event = ""

    # def on_enter(self, *args):
    #    global screen_to
    #    try:
    #        self.event = Clock.schedule_interval(self.check, 0.5)
    #        os.listdir(app + "TO_PROCESS")
    #        os.listdir(app + "PROCESSED")
    #    except Exception as e:
    #        screen_to = "pathimcam"
    #        self.manager.transition = SlideTransition(direction="down")
    #        self.manager.current = "no_folder_error"

    # def on_leave(self, *args):
    #    self.event.cancel()

    def on_enter(self, *args):
        global app
        global app_assets
        if getattr(sys, 'frozen', False):
            app = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))) + "/"
            app_assets = str(os.path.dirname(sys.executable)) + "/"
        elif __file__:
            app = str(os.path.dirname(__file__)) + "/"
            app_assets = str(os.path.dirname(__file__)) + "/"
        self.model = model_from_json(open(app_assets + "assets/neuralnet.json", "r").read())
        self.model.load_weights(app_assets + "assets/weights.h5")

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "choose"

    def scan(self, *args):
        try:
            image = cv2.VideoCapture(0)
            k, frame = image.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if switcher != "emotion":
                global classifier
                if switcher == "face":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                elif switcher == "body":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                elif switcher == "eye":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                elif switcher == "smile":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                if detection == ():
                    self.manager.transition = SlideTransition(direction="down")
                    self.manager.current = "nodetectionscam"
                else:
                    for (x, y, z, w) in detection:
                        cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                    cv2.imwrite(app + "/ProcessedImage=).png", frame)
                    image.release()
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "finalim"
            elif switcher == "emotion":
                face_recogn = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
                if detection == ():
                    self.manager.transition = SlideTransition(direction="down")
                    self.manager.current = "nodetectionscam"
                else:
                    for (x, y, z, w) in detection:
                        cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                        cut_gray = gray[y: y + z, x: x + w]
                        resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                        array_gray_im = ima.img_to_array(resized_cut_gray)
                        array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                        array_gray_im_expanded /= 255
                        prediction = self.model.predict(array_gray_im_expanded)
                        final_prediction = numpy.argmax(prediction[0])
                        list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
                        result = list_of_emotions[final_prediction]
                        if z >= 210 and w >= 210:
                            cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                        (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        (0, 0, 255), 1)
                    cv2.imwrite(app + "/ProcessedImage=).png", frame)
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "finalim"
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "errorimcam"

    def scan_area(self, *args):
        try:
            image = cv2.VideoCapture(0)
            k, frame = image.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if switcher != "emotion":
                global classifier
                if switcher == "face":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                elif switcher == "body":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                elif switcher == "eye":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                elif switcher == "smile":
                    classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                picker = 0
                if detection == ():
                    self.manager.transition = SlideTransition(direction="down")
                    self.manager.current = "nodetectionscam"
                else:
                    for (x, y, z, w) in detection:
                        picker += 1
                        image_fin = frame[y: y + z, x: x + w]
                        image_fin = cv2.resize(image_fin, (520, 400))
                        cv2.imwrite(app + "/ProcessedImage=)" + str(picker) + ".png", image_fin)
                    image.release()
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "finalim"
            elif switcher == "emotion":
                face_recogn = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
                if detection == ():
                    self.manager.transition = SlideTransition(direction="down")
                    self.manager.current = "nodetectionscam"
                else:
                    picker = 0
                    for (x, y, z, w) in detection:
                        picker += 1
                        image_f = frame[y: y + z, x: x + w]
                        image_final = cv2.resize(image_f, (520, 400))
                        cut_gray = gray[y: y + z, x: x + w]
                        resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                        array_gray_im = ima.img_to_array(resized_cut_gray)
                        array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                        array_gray_im_expanded /= 255
                        prediction = self.model.predict(array_gray_im_expanded)
                        final_prediction = numpy.argmax(prediction[0])
                        list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
                        result = list_of_emotions[final_prediction]
                        cv2.putText(image_final, result, (10, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.4,
                                    (0, 0, 255), 2)
                        cv2.imwrite(app + "/ProcessedImage=)" + str(picker) + ".png", image_final)
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "finalim"
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "errorimcam"

    def result(self, *args):
        image = cv2.VideoCapture(0)
        k, frame = image.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if switcher != "emotion":
            global classifier
            if switcher == "face":
                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
            elif switcher == "body":
                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
            elif switcher == "eye":
                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
            elif switcher == "smile":
                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
            detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            t = True
            counter = 0
            if detection == ():
                self.manager.transition = SlideTransition(direction="down")
                self.manager.current = "nodetectionscam"
            else:
                for (x, y, z, w) in detection:
                    counter += 1
                    cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                    if counter == 1:
                        cv2.putText(frame, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 15), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (20, 226, 20), 1)
                cv2.imshow("PREVIEW", frame)
                cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.waitKey(0)
                image.release()
                while t:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyAllWindows()
        elif switcher == "emotion":
            face_recogn = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
            detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
            t = True
            counter = 0
            if detection == ():
                self.manager.transition = SlideTransition(direction="down")
                self.manager.current = "nodetectionscam"
            else:
                for (x, y, z, w) in detection:
                    counter += 1
                    cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                    if counter == 1:
                        cv2.putText(frame, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (20, 226, 20), 1)
                    cut_gray = gray[y: y + z, x: x + w]
                    resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                    array_gray_im = ima.img_to_array(resized_cut_gray)
                    array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                    array_gray_im_expanded /= 255
                    prediction = self.model.predict(array_gray_im_expanded)
                    final_prediction = numpy.argmax(prediction[0])
                    list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
                    result = list_of_emotions[final_prediction]
                    if z >= 210 and w >= 210:
                        cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                    (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (0, 0, 255), 1)
                cv2.imshow("PREVIEW", frame)
                cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.waitKey(0)
                while t:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyAllWindows()

    def result_area(self, *args):
        image = cv2.VideoCapture(0)
        k, frame = image.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if switcher != "emotion":
            global classifier
            if switcher == "face":
                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
            elif switcher == "body":
                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
            elif switcher == "eye":
                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
            elif switcher == "smile":
                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
            detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            t = True
            counter = 0
            tracker = 0
            counter_of_images = []
            for element in detection:
                counter += 1
                counter_of_images.append(counter)
            if detection == ():
                self.manager.transition = SlideTransition(direction="down")
                self.manager.current = "nodetectionscam"
            else:
                for (x, y, z, w) in detection:
                    final_val = counter_of_images[-1]
                    value = counter_of_images[tracker]
                    tracker += 1
                    image_fin = frame[y: y + z, x: x + w]
                    image_fin = cv2.resize(image_fin, (520, 400))
                    if value == final_val:
                        cv2.putText(image_fin, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 11),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.30,
                                    (20, 226, 20), 1)
                    else:
                        cv2.putText(image_fin, "PRESS SPACE TO GET TO THE NEXT IMAGE", (10, 11),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.30,
                                    (20, 226, 20), 1)
                    cv2.imshow("PREVIEW", image_fin)
                    cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.waitKey(0)
                    image.release()
                while t:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyWindow("PREVIEW")
        elif switcher == "emotion":
            face_recogn = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
            detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
            if detection == ():
                self.manager.transition = SlideTransition(direction="down")
                self.manager.current = "nodetectionscam"
            else:
                t = True
                counter = 0
                tracker = 0
                counter_of_images = []
                for element in detection:
                    counter += 1
                    counter_of_images.append(counter)
                picker = 0
                for (x, y, z, w) in detection:
                    picker += 1
                    image_f = frame[y: y + z, x: x + w]
                    image_final = cv2.resize(image_f, (520, 400))
                    cut_gray = gray[y: y + z, x: x + w]
                    resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                    array_gray_im = ima.img_to_array(resized_cut_gray)
                    array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                    array_gray_im_expanded /= 255
                    prediction = self.model.predict(array_gray_im_expanded)
                    final_prediction = numpy.argmax(prediction[0])
                    list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
                    result = list_of_emotions[final_prediction]
                    cv2.putText(image_final, result, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.4,
                                (0, 0, 255), 2)
                    final_val = counter_of_images[-1]
                    value = counter_of_images[tracker]
                    tracker += 1
                    if value == final_val:
                        cv2.putText(image_final, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 11),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.30,
                                    (20, 226, 20), 1)
                    else:
                        cv2.putText(image_final, "PRESS SPACE TO GET TO THE NEXT IMAGE", (10, 11),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.30,
                                    (20, 226, 20), 1)
                    cv2.imshow("PREVIEW", image_final)
                    cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.waitKey(0)
                while t:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyWindow("PREVIEW")


class pathvid(Screen):
    def __init__(self, **kwargs):
        super(pathvid, self).__init__(**kwargs)
        self.layout = FloatLayout(size=(350, 600))
        self.popup_layout = FloatLayout(size=(175, 300))

        label4 = Label(text="VIDEO",
                       pos_hint={"x": 0.194, "y": 0.83},
                       color=(0.309, 0.933, 0.078, 4), size_hint=(0.62, .07))
        label3 = Label(text="WRITE FULL PATH TO A VIDEO (i.e. /Users/joe/video.mp4) OR PRESS FILE ICON TO CHOOSE FILE", pos_hint={"x": 0.01, "y": 0.24},
                       color=(0.309, 0.933, 0.078, 4))
        self.txt1 = TextInput(hint_text='PATH TO A VIDEO',
                              multiline=False,
                              size_hint=(0.55, .07),
                              pos_hint={'x': 0.2, 'y': 0.57},
                              background_color=(0.309, 0.933, 0.078, 4))
        but1 = Button(text="SAVE FULL VIDEO >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.33},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        but1.bind(on_press=self.scan)
        but3 = Button(text="SHOW RESULT", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.20, "y": 0.33},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        but3.bind(on_press=self.result)
        but4 = Button(text="SHOW ONLY DETECTED AREAS", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.08},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        but4.bind(on_press=self.result_area)
        but5 = Button(text="SAVE ONLY DETECTED AREAS >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.08},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        but5.bind(on_press=self.scan_area)
        but2 = Button(text="< BACK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.035, "y": 0.85},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.2, 0.1))
        but2.bind(on_press=self.switch)
        file_finder = Button(text="", background_normal=app_assets + "assets/file.png",
                             background_down=app_assets + "assets/file.png", pos_hint={"x": 0.75, "y": 0.56},
                             size_hint=(0.075, 0.095))
        file_finder.bind(on_press=self.pops)
        self.file_chooser = FileChooserIconView(dirselect=True)
        but_close_popup = Button(text="CANCEL", background_color=(0.309, 0.933, 0.078, 4),
                                 pos_hint={"x": 0.005, "y": 1.015},
                                 color=(0.141, 0.054, 0.078, 4), size_hint=(0.15, 0.05))
        but_ok_popup = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4),
                              pos_hint={"x": 0.84, "y": 1.015},
                              color=(0.141, 0.054, 0.078, 4), size_hint=(0.15, 0.05))
        but_close_popup.bind(on_press=self.close)
        but_ok_popup.bind(on_press=self.choose_file)
        self.layout.add_widget(label4)
        self.layout.add_widget(self.txt1)
        self.layout.add_widget(label3)
        self.layout.add_widget(but1)
        self.layout.add_widget(but2)
        self.layout.add_widget(but3)
        self.layout.add_widget(but4)
        self.layout.add_widget(but5)
        self.layout.add_widget(file_finder)
        self.add_widget(self.layout)
        self.model = ""
        self.popup_layout.add_widget(self.file_chooser)
        self.popup_layout.add_widget(but_close_popup)
        self.popup_layout.add_widget(but_ok_popup)
        self.popup = Popup(title="select your file", content=self.popup_layout)
        # self.event = ""

    # def on_enter(self, *args):
    #    global screen_to
    #    try:
    #        self.event = Clock.schedule_interval(self.check, 0.5)
    #        os.listdir(app + "TO_PROCESS")
    #        os.listdir(app + "PROCESSED")
    #    except Exception as e:
    #        screen_to = "pathimcam"
    #        self.manager.transition = SlideTransition(direction="down")
    #        self.manager.current = "no_folder_error"

    # def on_leave(self, *args):
    #    self.event.cancel()

    def on_enter(self, *args):
        global app
        global app_assets
        if getattr(sys, 'frozen', False):
            app = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))) + "/"
            app_assets = str(os.path.dirname(sys.executable)) + "/"
        elif __file__:
            app = str(os.path.dirname(__file__)) + "/"
            app_assets = str(os.path.dirname(__file__)) + "/"
        self.model = model_from_json(open(app_assets + "assets/neuralnet.json", "r").read())
        self.model.load_weights(app_assets + "assets/weights.h5")

    def pops(self, *args):
        self.popup.open()

    def close(self, *args):
        self.popup.dismiss()

    def choose_file(self, *args):
        try:
            self.txt1.text = self.file_chooser.selection[0]
            self.popup.dismiss()
        except Exception as e:
            self.txt1.text = ""
            self.popup.dismiss()

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "choose"

    def scan(self, *args):
        global txt1
        txt1 = str(self.txt1.text)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "loading_scan"

    def scan_area(self, *args):
        global txt1
        txt1 = str(self.txt1.text)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "loading_scan_areas"

    def result(self, *args):
        app = self.txt1.text
        try:
            if str(self.txt1.text[-3:]) != "peg" and str(self.txt1.text[-3:]) != "jpg" and str(
                    self.txt1.text[-3:]) != "gif" and str(
                self.txt1.text[-3:]) != "png" and str(self.txt1.text[-3:]) != "iff" and str(
                self.txt1.text[-3:]) != "psd" and str(
                self.txt1.text[-3:]) != "pdf" and str(self.txt1.text[-3:]) != "eps" and str(
                self.txt1.text[-3:]) != ".ai" and str(
                self.txt1.text[-3:]) != "ndd" and str(self.txt1.text[-3:]) != "raw":
                t = True
                video_check = cv2.VideoCapture(app)
                k, frame_raw_check = video_check.read()
                cv2.resize(frame_raw_check, (520, 400))
                video_check.release()
                video = cv2.VideoCapture(app)
                height_test = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
                width_test = video.get(cv2.CAP_PROP_FRAME_WIDTH)
                width = str(width_test)
                height = str(height_test)
                width = float(width)
                height = float(height)
                try:
                    if switcher != "emotion":
                        while t:
                            k, frame_raw = video.read()
                            frame = cv2.resize(frame_raw, (int(width), int(height)))
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            global classifier
                            if switcher == "face":
                                classifier = cv2.CascadeClassifier(
                                    app_assets + "assets/haarcascade_frontalface_default.xml")
                            elif switcher == "body":
                                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                            elif switcher == "eye":
                                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                            elif switcher == "smile":
                                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                            detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                            cv2.putText(frame, "PRESS SPACE ONCE OR TWICE TO CLOSE THE VIDEO", (10, 15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35,
                                        (20, 226, 20), 1)
                            for (x, y, z, w) in detection:
                                cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                            cv2.imshow("PREVIEW", frame)
                            cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                            cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            for event in pygame.event.get():
                                if event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_SPACE:
                                        t = False
                                        cv2.destroyWindow("PREVIEW")
                    elif switcher == "emotion":
                        t = True
                        while t:
                            k, frame_raw = video.read()
                            frame = cv2.resize(frame_raw, (int(width), int(height)))
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            face_recogn = cv2.CascadeClassifier(
                                app_assets + "assets/haarcascade_frontalface_default.xml")
                            detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
                            cv2.putText(frame, "PRESS SPACE TWICE TO CLOSE THE VIDEO", (10, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35,
                                        (20, 226, 20), 1)
                            for (x, y, z, w) in detection:
                                cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                                cut_gray = gray[y: y + z, x: x + w]
                                resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                                array_gray_im = ima.img_to_array(resized_cut_gray)
                                array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                                array_gray_im_expanded /= 255
                                prediction = self.model.predict(array_gray_im_expanded)
                                final_prediction = numpy.argmax(prediction[0])
                                list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised',
                                                    'neutral']
                                result = list_of_emotions[final_prediction]
                                if z >= 210 and w >= 210:
                                    cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX,
                                                1.4,
                                                (0, 0, 255), 2)
                                else:
                                    cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.35,
                                                (0, 0, 255), 1)
                            cv2.imshow("PREVIEW", frame)
                            cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                            cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            for event in pygame.event.get():
                                if event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_SPACE:
                                        t = False
                                        cv2.destroyAllWindows()
                except Exception as e:
                    video.release()
                    cv2.destroyAllWindows()
            else:
                self.manager.transition = SlideTransition(direction="down")
                self.manager.current = "errorvid"
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "errorvid"

    def result_area(self, *args):
        global txt1
        txt1 = str(self.txt1.text)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "loading_result_areas"


class pathvidcam(Screen):
    def __init__(self, **kwargs):
        super(pathvidcam, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        label1 = Label(text="VIDEO FROM YOUR CAMERA", pos_hint={"x": 0.01, "y": 0.25},
                       color=(0.309, 0.933, 0.078, 4))
        but1 = Button(text="RECORD AND SAVE FULL VIDEO >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        but1.bind(on_press=self.scan)
        but3 = Button(text="RECORD AND SHOW RESULT", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        but3.bind(on_press=self.result)
        but4 = Button(text="RECORD AND SHOW DETECTED AREAS", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        but4.bind(on_press=self.result_area)
        but5 = Button(text="RECORD AND SAVE DETECTED AREAS >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        but5.bind(on_press=self.scan_area)
        but2 = Button(text="< BACK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.035, "y": 0.85},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.2, 0.1))
        but2.bind(on_press=self.switch)

        layout.add_widget(label1)
        layout.add_widget(but1)
        layout.add_widget(but2)
        layout.add_widget(but3)
        layout.add_widget(but4)
        layout.add_widget(but5)
        self.add_widget(layout)
        self.model = ""
        # self.event = ""

    # def on_enter(self, *args):
    #    global screen_to
    #    try:
    #        self.event = Clock.schedule_interval(self.check, 0.5)
    #        os.listdir(app + "TO_PROCESS")
    #        os.listdir(app + "PROCESSED")
    #    except Exception as e:
    #        screen_to = "pathimcam"
    #        self.manager.transition = SlideTransition(direction="down")
    #        self.manager.current = "no_folder_error"

    # def on_leave(self, *args):
    #    self.event.cancel()

    def on_enter(self, *args):
        global app
        global app_assets
        if getattr(sys, 'frozen', False):
            app = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))) + "/"
            app_assets = str(os.path.dirname(sys.executable)) + "/"
        elif __file__:
            app = str(os.path.dirname(__file__)) + "/"
            app_assets = str(os.path.dirname(__file__)) + "/"
        self.model = model_from_json(open(app_assets + "assets/neuralnet.json", "r").read())
        self.model.load_weights(app_assets + "assets/weights.h5")

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "choose"

    def scan(self, *args):
        try:
            t = True
            video = cv2.VideoCapture(0)
            height_test = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width_test = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            width = str(width_test)
            height = str(height_test)
            width = float(width)
            height = float(height)
            capture = cv2.VideoWriter(app + "ProcessedVideo=).avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                      (int(width), int(height)))
            if switcher != "emotion":
                while t:
                    k, frame_raw = video.read()
                    frame = cv2.resize(frame_raw, (int(width), int(height)))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    global classifier
                    if switcher == "face":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                    elif switcher == "body":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                    elif switcher == "eye":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                    elif switcher == "smile":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                    detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                    for (x, y, z, w) in detection:
                        cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                    capture.write(frame)
                    cv2.putText(frame, "PRESS SPACE TO STOP RECORDING", (10, 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (20, 226, 20), 1)
                    cv2.imshow("RECORDING", frame)
                    cv2.namedWindow("RECORDING", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("RECORDING", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyAllWindows()
                                video.release()
                                self.manager.transition = SlideTransition(direction="left")
                                self.manager.current = "finalim"
            elif switcher == "emotion":
                while t:
                    k, frame_raw = video.read()
                    frame = cv2.resize(frame_raw, (int(width), int(height)))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_recogn = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                    detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
                    for (x, y, z, w) in detection:
                        cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                        cut_gray = gray[y: y + z, x: x + w]
                        resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                        array_gray_im = ima.img_to_array(resized_cut_gray)
                        array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                        array_gray_im_expanded /= 255
                        prediction = self.model.predict(array_gray_im_expanded)
                        final_prediction = numpy.argmax(prediction[0])
                        list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
                        result = list_of_emotions[final_prediction]
                        if z >= 210 and w >= 210:
                            cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                        (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        (0, 0, 255), 1)
                    capture.write(frame)
                    cv2.putText(frame, "PRESS SPACE TO STOP RECORDING", (10, 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (20, 226, 20), 1)
                    cv2.imshow("RECORDING", frame)
                    cv2.namedWindow("RECORDING", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("RECORDING", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyAllWindows()
                                video.release()
                                self.manager.transition = SlideTransition(direction="left")
                                self.manager.current = "finalim"
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "errorvidcam"

    def scan_area(self, *args):
        try:
            capture = cv2.VideoWriter(app + "ProcessedVideo=).avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                      (520, 400))
            t = True
            video = cv2.VideoCapture(0)
            if switcher != "emotion":
                while t:
                    k, frame_raw = video.read()
                    frame = cv2.resize(frame_raw, (520, 400))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    global classifier
                    if switcher == "face":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                    elif switcher == "body":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                    elif switcher == "eye":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                    elif switcher == "smile":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                    detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                    if detection == ():
                        capture.write(frame)
                        cv2.putText(frame, "PRESS SPACE ONCE OR TWICE TO STOP RECORDING", (10, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (20, 226, 20), 1)
                        cv2.putText(frame, "NO DETECTIONS", (200, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (20, 226, 20), 1)
                        cv2.imshow("RECORDING", frame)
                        cv2.namedWindow("RECORDING", cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty("RECORDING", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        for (x, y, z, w) in detection:
                            cut = frame[y:y + z, x:x + w]
                            cut_image = cv2.resize(cut, (520, 400))
                            capture.write(cut_image)
                            cv2.putText(cut_image, "PRESS SPACE ONCE OR TWICE TO STOP RECORDING", (10, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35,
                                        (20, 226, 20), 1)
                            cv2.imshow("RECORDING", cut_image)
                            cv2.namedWindow("RECORDING", cv2.WND_PROP_FULLSCREEN)
                            cv2.setWindowProperty("RECORDING", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyAllWindows()
                                video.release()
                                self.manager.transition = SlideTransition(direction="left")
                                self.manager.current = "finalim"
            elif switcher == "emotion":
                while t:
                    k, frame_raw = video.read()
                    frame = cv2.resize(frame_raw, (520, 400))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_recogn = cv2.CascadeClassifier(
                        app_assets + "assets/haarcascade_frontalface_default.xml")
                    detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
                    if detection == ():
                        capture.write(frame)
                        cv2.putText(frame, "PRESS SPACE ONCE OR TWICE TO STOP RECORDING", (10, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (20, 226, 20), 1)
                        cv2.putText(frame, "NO DETECTIONS", (200, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (20, 226, 20), 1)
                    else:
                        for (x, y, z, w) in detection:
                            image_f = frame[y: y + z, x: x + w]
                            image_final = cv2.resize(image_f, (520, 400))
                            cut_gray = gray[y: y + z, x: x + w]
                            resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                            array_gray_im = ima.img_to_array(resized_cut_gray)
                            array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                            array_gray_im_expanded /= 255
                            prediction = self.model.predict(array_gray_im_expanded)
                            final_prediction = numpy.argmax(prediction[0])
                            list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised',
                                                'neutral']
                            result = list_of_emotions[final_prediction]
                            cv2.putText(image_final, result, (10, 45),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.4,
                                        (0, 0, 255), 2)
                            capture.write(image_final)
                            cv2.putText(image_final, "PRESS SPACE ONCE OR TWICE TO STOP RECORDING", (10, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35,
                                        (20, 226, 20), 1)
                            frame = image_final
                    cv2.imshow("RECORDING", frame)
                    cv2.namedWindow("RECORDING", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("RECORDING", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyAllWindows()
                                video.release()
                                self.manager.transition = SlideTransition(direction="left")
                                self.manager.current = "finalim"
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "errorvidcam"

    def result(self, *args):
        try:
            t = True
            video = cv2.VideoCapture(0)
            height_test = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width_test = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            width = str(width_test)
            height = str(height_test)
            width = float(width)
            height = float(height)
            if switcher != "emotion":
                while t:
                    k, frame_raw = video.read()
                    frame = cv2.resize(frame_raw, (int(width), int(height)))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    global classifier
                    if switcher == "face":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                    elif switcher == "body":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                    elif switcher == "eye":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                    elif switcher == "smile":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                    detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                    if detection == ():
                        cv2.putText(frame, "NO DETECTIONS", (200, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (20, 226, 20), 1)
                    else:
                        for (x, y, z, w) in detection:
                            cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                    cv2.putText(frame, "PRESS SPACE ONCE/TWICE TO STOP SHOWING VIDEO", (10, 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (20, 226, 20), 1)
                    cv2.imshow("PREVIEW", frame)
                    cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyAllWindows()
                                video.release()
            elif switcher == "emotion":
                t = True
                while t:
                    k, frame_raw = video.read()
                    frame = cv2.resize(frame_raw, (int(width), int(height)))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_recogn = cv2.CascadeClassifier(
                        app_assets + "assets/haarcascade_frontalface_default.xml")
                    detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
                    if detection == ():
                        cv2.putText(frame, "NO DETECTIONS", (200, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (20, 226, 20), 1)
                        for (x, y, z, w) in detection:
                            cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                            cv2.putText(frame, "PRESS SPACE ONCE/TWICE TO STOP SHOWING VIDEO", (10, 15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35,
                                        (20, 226, 20), 1)
                    else:
                        cv2.putText(frame, "PRESS SPACE ONCE/TWICE TO STOP SHOWING VIDEO", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (20, 226, 20), 1)
                        for (x, y, z, w) in detection:
                            cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                            cut_gray = gray[y: y + z, x: x + w]
                            resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                            array_gray_im = ima.img_to_array(resized_cut_gray)
                            array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                            array_gray_im_expanded /= 255
                            prediction = self.model.predict(array_gray_im_expanded)
                            final_prediction = numpy.argmax(prediction[0])
                            list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised',
                                                'neutral']
                            result = list_of_emotions[final_prediction]
                            if z >= 210 and w >= 210:
                                cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX,
                                            1.4,
                                            (0, 0, 255), 2)
                            else:
                                cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.35,
                                            (0, 0, 255), 1)
                    cv2.imshow("PREVIEW", frame)
                    cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyAllWindows()
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "errorvidcam"

    def result_area(self, *args):
        try:
            t = True
            video = cv2.VideoCapture(0)
            if switcher != "emotion":
                while t:
                    k, frame_raw = video.read()
                    frame = cv2.resize(frame_raw, (520, 400))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    global classifier
                    if switcher == "face":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                    elif switcher == "body":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                    elif switcher == "eye":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                    elif switcher == "smile":
                        classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                    detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                    if detection == ():
                        cv2.putText(frame, "PRESS SPACE ONCE OR TWICE TO STOP SHOWING VIDEO", (10, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (20, 226, 20), 1)
                        cv2.putText(frame, "NO DETECTIONS", (140, 205),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (20, 226, 20), 1)
                    else:
                        for (x, y, z, w) in detection:
                            cut = frame[y:y + z, x:x + w]
                            cut_image = cv2.resize(cut, (520, 400))
                            cv2.putText(cut_image, "PRESS SPACE ONCE OR TWICE TO STOP SHOWING VIDEO", (10, 15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35,
                                        (20, 226, 20), 1)
                            frame = cut_image
                    cv2.imshow("PREVIEW", frame)
                    cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyAllWindows()
                                video.release()
            elif switcher == "emotion":
                while t:
                    k, frame_raw = video.read()
                    frame = cv2.resize(frame_raw, (520, 400))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_recogn = cv2.CascadeClassifier(
                        app_assets + "assets/haarcascade_frontalface_default.xml")
                    detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
                    if detection == ():
                        cv2.putText(frame, "PRESS SPACE ONCE OR TWICE TO STOP SHOWING VIDEO", (10, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (20, 226, 20), 1)
                        cv2.putText(frame, "NO DETECTIONS", (140, 205),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (20, 226, 20), 1)
                    else:
                        for (x, y, z, w) in detection:
                            image_f = frame[y: y + z, x: x + w]
                            image_final = cv2.resize(image_f, (520, 400))
                            cut_gray = gray[y: y + z, x: x + w]
                            resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                            array_gray_im = ima.img_to_array(resized_cut_gray)
                            array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                            array_gray_im_expanded /= 255
                            prediction = self.model.predict(array_gray_im_expanded)
                            final_prediction = numpy.argmax(prediction[0])
                            list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised',
                                                'neutral']
                            result = list_of_emotions[final_prediction]
                            cv2.putText(image_final, result, (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.4,
                                        (0, 0, 255), 2)
                            cv2.putText(image_final, "PRESS SPACE TWICE TO CLOSE THE VIDEO", (10, 11),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.30,
                                        (20, 226, 20), 1)
                            frame = image_final
                    cv2.imshow("PREVIEW", frame)
                    cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                t = False
                                cv2.destroyAllWindows()
                                video.release()
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "errorvidcam"


class finalim(Screen):
    def __init__(self, **kwargs):
        super(finalim, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        label1 = Label(
            text="$$$ YOUR PROCESSED FILE HAS BEEN SAVED IN THE FOLDER PROCESSED (located in the same folder where your app is) $$$",
            pos_hint={"x": 0, "y": 0.05}, color=(0.309, 0.933, 0.078, 4), font_size="12.5sp")
        but1 = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.30, "y": 0.32},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.4, 0.10))
        but1.bind(on_press=self.switch)
        img = Image(source=app_assets + "assets/logo2.png", size_hint=(0.15, .3), pos_hint={"x": 0.423, "y": 0.65})

        layout.add_widget(label1)
        layout.add_widget(but1)
        layout.add_widget(img)
        self.add_widget(layout)

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "welcome"


class nodetections(Screen):
    def __init__(self, **kwargs):
        super(nodetections, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        label1 = Label(text="NO DETECTIONS",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="16sp")
        but1 = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.35, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.10))
        but1.bind(on_press=self.switch)

        layout.add_widget(label1)
        layout.add_widget(but1)
        self.add_widget(layout)

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "pathim"


class nodetectionscam(Screen):
    def __init__(self, **kwargs):
        super(nodetectionscam, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        label1 = Label(text="NO DETECTIONS",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="16sp")
        but1 = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.35, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.10))
        but1.bind(on_press=self.switch)

        layout.add_widget(label1)
        layout.add_widget(but1)
        self.add_widget(layout)

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "pathimcam"


class errorimcam(Screen):
    def __init__(self, **kwargs):
        super(errorimcam, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        label1 = Label(text="ERROR: EITHER NAME/TYPE OF YOUR FOLDER/FILE IS WRONG OR FOLDER/FILE DOESN'T EXIST",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="13sp")
        but1 = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.30, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.4, 0.10))
        but1.bind(on_press=self.switch)

        layout.add_widget(label1)
        layout.add_widget(but1)
        self.add_widget(layout)

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "pathimcam"


class errorvid(Screen):
    def __init__(self, **kwargs):
        super(errorvid, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        label1 = Label(text="ERROR: EITHER NAME/TYPE OF YOUR FOLDER/FILE IS WRONG OR FOLDER/FILE DOESN'T EXIST",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="13sp")
        but1 = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.30, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.4, 0.10))
        but1.bind(on_press=self.switch)

        layout.add_widget(label1)
        layout.add_widget(but1)
        self.add_widget(layout)

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "pathvid"


class errorvidcam(Screen):
    def __init__(self, **kwargs):
        super(errorvidcam, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        label1 = Label(text="ERROR: EITHER NAME/TYPE OF YOUR FOLDER/FILE IS WRONG OR FOLDER/FILE DOESN'T EXIST",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="13sp")
        but1 = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.30, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.4, 0.10))
        but1.bind(on_press=self.switch)

        layout.add_widget(label1)
        layout.add_widget(but1)
        self.add_widget(layout)

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "pathvidcam"


class loading_scan(Screen):
    def __init__(self, **kwargs):
        super(loading_scan, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        wimg = AsyncImage(source=app_assets + 'assets/loading.gif',
                          size_hint=(0.15, 1 / 4),
                          keep_ratio=False,
                          allow_stretch=True,
                          pos_hint={'x': 0.435, 'y': 0.4})
        label1 = Label(text="PROCESSING", pos_hint={"x": 0.005, "y": 0.20}, color=(0.309, 0.933, 0.078, 4))
        label2 = Label(text="30 seconds of video get processed approximately for 1 minute 15 seconds",
                       pos_hint={"x": 0.005, "y": -0.3}, color=(0.309, 0.933, 0.078, 4))
        layout.add_widget(wimg)
        layout.add_widget(label1)
        layout.add_widget(label2)
        self.add_widget(layout)
        self.model = ""

    def on_enter(self, *args):
        try:
            self.model = model_from_json(open(app_assets + "assets/neuralnet.json", "r").read())
            self.model.load_weights(app_assets + "assets/weights.h5")
            audio = AudioFileClip(txt1)
            audio.write_audiofile(os.path.dirname(txt1) + "/audio.mp3")
            t = True
            video = cv2.VideoCapture(txt1)
            height_test = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width_test = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            width = str(width_test)
            height = str(height_test)
            width = float(width)
            height = float(height)
            capture = cv2.VideoWriter(os.path.dirname(txt1) + "/ProcessedVideo=).avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                      (int(width), int(height)))
            try:
                if switcher != "emotion":
                    while t:
                        k, frame_raw = video.read()
                        frame = cv2.resize(frame_raw, (int(width), int(height)))
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        global classifier
                        if switcher == "face":
                            classifier = cv2.CascadeClassifier(
                                app_assets + "assets/haarcascade_frontalface_default.xml")
                        elif switcher == "body":
                            classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                        elif switcher == "eye":
                            classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                        elif switcher == "smile":
                            classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                        detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                        for (x, y, z, w) in detection:
                            cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                        capture.write(frame)
                elif switcher == "emotion":
                    while t:
                        k, frame_raw = video.read()
                        frame = cv2.resize(frame_raw, (int(width), int(height)))
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        face_recogn = cv2.CascadeClassifier(app_assets + "assets/haarcascade_frontalface_default.xml")
                        detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
                        for (x, y, z, w) in detection:
                            cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                            cut_gray = gray[y: y + z, x: x + w]
                            resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                            array_gray_im = ima.img_to_array(resized_cut_gray)
                            array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                            array_gray_im_expanded /= 255
                            prediction = self.model.predict(array_gray_im_expanded)
                            final_prediction = numpy.argmax(prediction[0])
                            list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
                            result = list_of_emotions[final_prediction]
                            if z >= 210 and w >= 210:
                                cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                            (0, 0, 255), 2)
                            else:
                                cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                            (0, 0, 255), 1)
                        capture.write(frame)
            except Exception as e:
                video.release()
                # input_audio = ffmpeg.input(app_assets + "assets/audio.mp3")
                # input_video = ffmpeg.input(app_assets + "assets/video.avi")
                # (
                #    ffmpeg
                #        .concat(input_video, input_audio, v=1, a=1)
                #       .output(app + "PROCESSED/ProcessedVideo=).avi")
                #        .global_args('-loglevel', 'quiet')
                #        .run(capture_stdout=True, overwrite_output=True)
                # )
                self.manager.transition = SlideTransition(direction="left")
                self.manager.current = "finalim"
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "errorvid"


class loading_scan_areas(Screen):
    def __init__(self, **kwargs):
        super(loading_scan_areas, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        wimg = AsyncImage(source=app_assets + 'assets/loading.gif',
                          size_hint=(0.15, 1 / 4),
                          keep_ratio=False,
                          allow_stretch=True,
                          pos_hint={'x': 0.435, 'y': 0.4})
        label1 = Label(text="PROCESSING", pos_hint={"x": 0.005, "y": 0.20}, color=(0.309, 0.933, 0.078, 4))
        label2 = Label(text="30 seconds of video get processed approximately for 1 minute 15 seconds",
                       pos_hint={"x": 0.005, "y": -0.3}, color=(0.309, 0.933, 0.078, 4))
        layout.add_widget(wimg)
        layout.add_widget(label1)
        layout.add_widget(label2)
        self.add_widget(layout)
        self.model = ""

    def on_enter(self, *args):
        try:
            self.model = model_from_json(open(app_assets + "assets/neuralnet.json", "r").read())
            self.model.load_weights(app_assets + "assets/weights.h5")
            if str(txt1[-3:]) != "peg" and str(txt1[-3:]) != "jpg" and str(
                    txt1[-3:]) != "gif" and str(txt1[-3:]) != "png" and str(
                txt1[-3:]) != "iff" and str(txt1[-3:]) != "psd" and str(
                txt1[-3:]) != "pdf" and str(txt1[-3:]) != "eps" and str(txt1[-3:]) != ".ai" and str(
                txt1[-3:]) != "ndd" and str(txt1[-3:]) != "raw":
                t = True
                video_check = cv2.VideoCapture(txt1)
                k, frame_raw_check = video_check.read()
                cv2.resize(frame_raw_check, (520, 400))
                video_check.release()
                video = cv2.VideoCapture(txt1)
                capture = cv2.VideoWriter(os.path.dirname(txt1) + "/ProcessedVideo=).avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                          (520, 400))
                try:
                    if switcher != "emotion":
                        while t:
                            k, frame_raw = video.read()
                            frame = cv2.resize(frame_raw, (520, 400))
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            global classifier
                            if switcher == "face":
                                classifier = cv2.CascadeClassifier(
                                    app_assets + "assets/haarcascade_frontalface_default.xml")
                            elif switcher == "body":
                                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                            elif switcher == "eye":
                                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                            elif switcher == "smile":
                                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                            detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                            for (x, y, z, w) in detection:
                                final_frame = frame[y:y + z, x:x + w]
                                final_cut_frame = cv2.resize(final_frame, (520, 400))
                                capture.write(final_cut_frame)
                    elif switcher == "emotion":
                        while t:
                            k, frame_raw = video.read()
                            frame = cv2.resize(frame_raw, (520, 400))
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            face_recogn = cv2.CascadeClassifier(
                                app_assets + "assets/haarcascade_frontalface_default.xml")
                            detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
                            picker = 0
                            for (x, y, z, w) in detection:
                                picker += 1
                                image_f = frame[y: y + z, x: x + w]
                                image_final = cv2.resize(image_f, (520, 400))
                                cut_gray = gray[y: y + z, x: x + w]
                                resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                                array_gray_im = ima.img_to_array(resized_cut_gray)
                                array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                                array_gray_im_expanded /= 255
                                prediction = self.model.predict(array_gray_im_expanded)
                                final_prediction = numpy.argmax(prediction[0])
                                list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised',
                                                    'neutral']
                                result = list_of_emotions[final_prediction]
                                cv2.putText(image_final, result, (10, 45),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1.4,
                                            (0, 0, 255), 2)
                                capture.write(image_final)
                except Exception as e:
                    video.release()
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "finalim"
            else:
                self.manager.transition = SlideTransition(direction="down")
                self.manager.current = "errorvid"
        except Exception as e:
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "errorvid"


class loading_result_areas(Screen):
    def __init__(self, **kwargs):
        super(loading_result_areas, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        wimg = AsyncImage(source=app_assets + 'assets/loading.gif',
                          size_hint=(0.15, 1 / 4),
                          keep_ratio=False,
                          allow_stretch=True,
                          pos_hint={'x': 0.435, 'y': 0.4})
        label1 = Label(text="PROCESSING", pos_hint={"x": 0.005, "y": 0.20}, color=(0.309, 0.933, 0.078, 4))
        label2 = Label(text="30 seconds of video get processed approximately for 1 minute 15 seconds",
                       pos_hint={"x": 0.005, "y": -0.3}, color=(0.309, 0.933, 0.078, 4))
        layout.add_widget(wimg)
        layout.add_widget(label1)
        layout.add_widget(label2)
        self.add_widget(layout)
        self.switcher = True
        self.model = ""

    def on_enter(self, *args):
        try:
            self.model = model_from_json(open(app_assets + "assets/neuralnet.json", "r").read())
            self.model.load_weights(app_assets + "assets/weights.h5")
            if str(txt1[-3:]) != "peg" and str(txt1[-3:]) != "jpg" and str(txt1[-3:]) != "gif" and str(
                    txt1[-3:]) != "png" and str(txt1[-3:]) != "iff" and str(txt1[-3:]) != "psd" and str(
                txt1[-3:]) != "pdf" and str(txt1[-3:]) != "eps" and str(txt1[-3:]) != ".ai" and str(
                txt1[-3:]) != "ndd" and str(txt1[-3:]) != "raw":
                t = True
                video_check = cv2.VideoCapture(txt1)
                k, frame_raw_check = video_check.read()
                cv2.resize(frame_raw_check, (520, 400))
                video_check.release()
                video = cv2.VideoCapture(txt1)
                capture = cv2.VideoWriter(app_assets + "assets/detected.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                          (520, 400))
                try:
                    if switcher != "emotion":
                        while t:
                            k, frame_raw = video.read()
                            frame = cv2.resize(frame_raw, (520, 400))
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            global classifier
                            if switcher == "face":
                                classifier = cv2.CascadeClassifier(
                                    app_assets + "assets/haarcascade_frontalface_default.xml")
                            elif switcher == "body":
                                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_fullbody.xml")
                            elif switcher == "eye":
                                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_eye.xml")
                            elif switcher == "smile":
                                classifier = cv2.CascadeClassifier(app_assets + "assets/haarcascade_smile.xml")
                            detection = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                            for (x, y, z, w) in detection:
                                final_frame = frame[y:y + z, x:x + w]
                                final_cut_frame = cv2.resize(final_frame, (520, 400))
                                cv2.putText(final_cut_frame, "PRESS SPACE ONCE OR TWICE TO CLOSE THE VIDEO", (10, 11),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.30,
                                            (20, 226, 20), 1)
                                capture.write(final_cut_frame)
                    elif switcher == "emotion":
                        t = True
                        while t:
                            k, frame_raw = video.read()
                            frame = cv2.resize(frame_raw, (520, 400))
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            face_recogn = cv2.CascadeClassifier(
                                app_assets + "assets/haarcascade_frontalface_default.xml")
                            detection = face_recogn.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
                            for (x, y, z, w) in detection:
                                image_f = frame[y: y + z, x: x + w]
                                image_final = cv2.resize(image_f, (520, 400))
                                cut_gray = gray[y: y + z, x: x + w]
                                resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                                array_gray_im = ima.img_to_array(resized_cut_gray)
                                array_gray_im_expanded = numpy.expand_dims(array_gray_im, axis=0)
                                array_gray_im_expanded /= 255
                                prediction = self.model.predict(array_gray_im_expanded)
                                final_prediction = numpy.argmax(prediction[0])
                                list_of_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised',
                                                    'neutral']
                                result = list_of_emotions[final_prediction]
                                cv2.putText(image_final, result, (10, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1.4,
                                            (0, 0, 255), 2)
                                cv2.putText(image_final, "PRESS SPACE TWICE TO CLOSE THE VIDEO", (10, 11),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.30,
                                            (20, 226, 20), 1)
                                capture.write(image_final)
                                for event in pygame.event.get():
                                    if event.type == pygame.KEYDOWN:
                                        if event.key == pygame.K_SPACE:
                                            t = False
                                            cv2.destroyWindow("PREVIEW")
                except Exception as e:
                    video.release()
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "pathvid"
            else:
                self.switcher = False
                self.manager.transition = SlideTransition(direction="down")
                self.manager.current = "errorvid"
        except Exception as e:
            self.switcher = False
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "errorvid"

    def on_leave(self, *args):
        if self.switcher:
            detected = ""
            try:
                fin = True
                detected = cv2.VideoCapture(app_assets + "assets/detected.avi")
                while fin:
                    kk, frame_fin = detected.read()
                    cv2.imshow("PREVIEW", frame_fin)
                    cv2.namedWindow("PREVIEW", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("PREVIEW", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    time.sleep(0.1)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                fin = False
                                cv2.destroyAllWindows()
            except Exception as e:
                detected.release()
                cv2.destroyAllWindows()


class no_folder_error(Screen):
    def __init__(self, **kwargs):
        super(no_folder_error, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        label1 = Label(text='PLEASE PLACE "TO_PROCESS" AND "PROCESSED" IN THE SAME FOLDER WHERE "SquareFace" APP IS',
                       pos_hint={"x": 0, "y": 0.065}, color=(0.309, 0.933, 0.078, 4), font_size="15sp")
        layout.add_widget(label1)
        self.add_widget(layout)
        self.event = ""

    def on_enter(self, *args):
        self.event = Clock.schedule_interval(self.check, 0.5)

    def check(self, *args):
        try:
            global app
            global app_assets
            if getattr(sys, 'frozen', False):
                app = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))) + "/"
                app_assets = str(os.path.dirname(sys.executable)) + "/"
            elif __file__:
                app = str(os.path.dirname(__file__)) + "/"
                app_assets = str(os.path.dirname(__file__)) + "/"
            os.listdir(app + "TO_PROCESS")
            os.listdir(app + "PROCESSED")
            self.event.cancel()
            self.manager.transition = SlideTransition(direction="up")
            self.manager.current = screen_to
        except Exception as e:
            pass


class error(Screen):
    def __init__(self, **kwargs):
        super(error, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        label1 = Label(text="ERROR: EITHER NAME/TYPE OF YOUR FOLDER/FILE IS WRONG OR FOLDER/FILE DOESN'T EXIST",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="13sp")
        but1 = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.30, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.4, 0.10))
        but1.bind(on_press=self.switch)

        layout.add_widget(label1)
        layout.add_widget(but1)
        self.add_widget(layout)

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "pathim"


class premium(Screen):

    def __init__(self, **kwargs):
        super(premium, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        # label1 = Label(text="THIS OPTION IS AVAILABLE ONLY IN PREMIUM VERSION", pos_hint={"x": 0.01, "y": 0.15},
        # color=(0.309, 0.933, 0.078, 4))
        # but1 = Button(text="BUY PREMIUM", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.36, "y": 0.36},
        # color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.20))
        label1 = Label(text="COMING SOON", pos_hint={"x": 0.01, "y": 0.15},
                       color=(0.309, 0.933, 0.078, 4))
        but1 = Button(text="COMING SOON", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.36, "y": 0.36},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.20))
        but2 = Button(text="< BACK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.035, "y": 0.85},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.2, 0.1))
        but2.bind(on_press=self.switch)

        layout.add_widget(label1)
        layout.add_widget(but1)
        layout.add_widget(but2)
        self.add_widget(layout)

    def switch(self, *args):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "welcome"


Manager = ScreenManager()
Manager.add_widget(welcome(name="welcome"))
Manager.add_widget(premium(name="premium"))
Manager.add_widget(choose(name="choose"))
Manager.add_widget(pathim(name="pathim"))
Manager.add_widget(finalim(name="finalim"))
Manager.add_widget(pathimcam(name="pathimcam"))
Manager.add_widget(pathvid(name="pathvid"))
Manager.add_widget(pathvidcam(name="pathvidcam"))
Manager.add_widget(nodetections(name="nodetections"))
Manager.add_widget(nodetectionscam(name="nodetectionscam"))
Manager.add_widget(loading_scan(name="loading_scan"))
Manager.add_widget(loading_scan_areas(name="loading_scan_areas"))
Manager.add_widget(loading_result_areas(name="loading_result_areas"))
Manager.add_widget(errorimcam(name="errorimcam"))
Manager.add_widget(errorvid(name="errorvid"))
Manager.add_widget(errorvidcam(name="errorvidcam"))
Manager.add_widget(error(name="error"))
Manager.add_widget(no_folder_error(name="no_folder_error"))


class SquareFace(App):
    def build(self):
        return Manager


def file():
    global app
    global app_assets
    if getattr(sys, 'frozen', False):
        app = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))) + "/"
        app_assets = str(os.path.dirname(sys.executable)) + "/"
    elif __file__:
        app = str(os.path.dirname(__file__)) + "/"
        app_assets = str(os.path.dirname(__file__)) + "/"
    pygame.init()
    Config.set("kivy", "window_icon", app_assets + "assets/logo.png")
    Config.set('graphics', 'resizable', False)
    Config.set("graphics", "width", "850")
    Config.set("graphics", "height", "630")
    Config.write()


if __name__ == "__main__":
    file()
    SquareFace().run()
