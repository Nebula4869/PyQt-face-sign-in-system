from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageDraw, ImageFont
from tensorflow.python.platform import gfile
from core import detect_face
import tensorflow as tf
import numpy as np
import qtawesome
import imageio
import time
import cv2
import sys
import os
import gc


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        """控件"""
        self.button_switch = QtWidgets.QPushButton(qtawesome.icon('fa.camera', color='black'), '启动')
        self.button_sign = QtWidgets.QPushButton(qtawesome.icon('fa.calendar-check-o', color='black'), '签到')
        self.button_register = QtWidgets.QPushButton(qtawesome.icon('fa.sign-in', color='black'), '注册')
        self.button_quit = QtWidgets.QPushButton(qtawesome.icon('fa.power-off', color='black'), '退出')
        self.button_admin = QtWidgets.QPushButton(qtawesome.icon('fa.puzzle-piece', color='black'), '管理模式')
        self.button_password = QtWidgets.QPushButton(qtawesome.icon('fa.edit', color='black'), '修改密码')
        self.button_import = QtWidgets.QPushButton(qtawesome.icon('fa.user-plus', color='black'), '导入用户')
        self.button_delete = QtWidgets.QPushButton(qtawesome.icon('fa.user-times', color='black'), '删除用户')
        self.label_show_camera = QtWidgets.QLabel()
        self.label_show_logo = QtWidgets.QLabel()
        self.label_show_face = QtWidgets.QLabel()
        self.show_record = QtWidgets.QTextEdit()
        img = imageio.imread('./resources/user.png')
        self.face_background = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGBA8888)
        self.set_slot()
        self.set_ui()
        """管理模式"""
        self.admin_mode = False
        self.admin_password = '123456'
        """摄像头画面捕获"""
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.show_camera)
        self.cap = cv2.VideoCapture()
        self.camera_id = 0
        """初始化MTCNN"""
        self.pnet, self.rnet, self.onet = self.init_mtcnn()
        """初始化FaceNet"""
        self.sess = tf.Session()
        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        graph_def.ParseFromString(gfile.FastGFile('./core/facenet.pb', 'rb').read())
        tf.import_graph_def(graph_def, name='graph')
        self.net_inputs = tf.get_default_graph().get_tensor_by_name('graph/input:0')
        self.net_outputs = tf.get_default_graph().get_tensor_by_name('graph/embeddings:0')
        self.phase_train = tf.get_default_graph().get_tensor_by_name('graph/phase_train:0')
        """加载人脸库"""
        self.database_dir = './faces/'  # 人脸库根目录
        if not os.path.exists(self.database_dir):
            os.makedirs(self.database_dir)
        self.known_names, self.known_embeddings = self.load_faces()  # 人脸库中的用户姓名与特征向量
        """人脸检测和识别的成员变量"""
        self.face_size = 1  # 人脸检测的最小尺寸
        self.scale_factor = 1  # 缩放尺度因子
        self.tolerance = 0.8  # 人脸识别阈值
        self.face_buffer = None  # 存储实时捕捉到的人脸
        self.record = open('签到记录.txt', 'a')

    def set_slot(self):
        """连接槽函数"""
        self.button_switch.clicked.connect(self.camera_switch)
        self.button_sign.clicked.connect(self.sign_in)
        self.button_register.clicked.connect(self.register_user)
        self.button_quit.clicked.connect(self.quit)
        self.button_admin.clicked.connect(self.admin_switch)
        self.button_password.clicked.connect(self.change_password)
        self.button_import.clicked.connect(self.import_user)
        self.button_delete.clicked.connect(self.delete_user)

    def set_ui(self):
        """界面美化"""
        for button in (self.button_switch, self.button_sign, self.button_register, self.button_quit,
                       self.button_admin, self.button_password, self.button_import, self.button_delete):
            button.setStyleSheet("QPushButton{border-image: url(resources/btn.png);"
                                 "background:#FFFFF;"
                                 "color:black;}"
                                 "QPushButton:hover{border-image: url(resources/btn_hover.png)}"
                                 "QPushButton:pressed{border-image: url(resources/btn_hover.png);}")
            button.setFixedSize(100, 30)

        self.label_show_camera.setFixedSize(1280, 720)
        self.label_show_camera.setStyleSheet("QLabel{background:#EFEFEF;"
                                             "border:5px solid #efb852;"
                                             "border-radius:0px;}")

        self.label_show_logo.setFixedSize(435, 160)
        img = imageio.imread('./resources/logo.png')
        logo = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGBA8888)
        self.label_show_logo.setPixmap(QtGui.QPixmap.fromImage(logo))
        self.label_show_logo.setStyleSheet("QLabel{border:0px;}")

        self.label_show_face.setFixedSize(160, 160)
        self.label_show_face.setStyleSheet("QLabel{background:#FFFFFF;"
                                           "border-radius:0px;}")
        self.label_show_face.setPixmap(QtGui.QPixmap.fromImage(self.face_background))

        self.show_record.setMaximumWidth(400)
        self.show_record.setReadOnly(True)
        self.show_record.setFontPointSize(12)
        self.show_record.setFontFamily('黑体')
        self.show_record.setStyleSheet("QTextEdit{background:#EFEFEF;"
                                       "border:1px solid #efb852;}")

        buttons_layout = QtWidgets.QGridLayout()
        buttons_layout.addWidget(self.button_switch, 0, 0, 1, 1)
        buttons_layout.addWidget(self.button_sign, 1, 0, 1, 1)
        buttons_layout.addWidget(self.button_register, 2, 0, 1, 1)
        buttons_layout.addWidget(self.button_quit, 3, 0, 1, 1)
        buttons_layout.addWidget(self.button_admin, 0, 1, 1, 1)
        buttons_layout.addWidget(self.button_password, 1, 1, 1, 1)
        buttons_layout.addWidget(self.button_import, 2, 1, 1, 1)
        buttons_layout.addWidget(self.button_delete, 3, 1, 1, 1)

        panel_layout = QtWidgets.QHBoxLayout()
        panel_layout.addWidget(self.label_show_logo)
        panel_layout.addLayout(buttons_layout)
        panel_layout.addWidget(self.label_show_face)
        panel_layout.addWidget(self.show_record)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.label_show_camera)
        main_layout.addLayout(panel_layout)

        widget = QtWidgets.QWidget()
        widget.setStyleSheet("QWidget{color:#000000;"
                             "background:#FFFFFF;"
                             "border:1px solid #CFCFCF;"
                             "border-radius:10px;}")
        widget.setLayout(main_layout)

        self.setCentralWidget(widget)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.move((1920 - 1298) // 2, (1080 - 904) // 2)

    @staticmethod
    def init_mtcnn():
        """初始化MTCNN"""
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        return pnet, rnet, onet

    def load_faces(self):
        """加载人脸库"""
        face_list = []
        face_names = []
        for face in os.listdir(self.database_dir):
            img = imageio.imread(os.path.join(self.database_dir, face))
            name = os.path.basename(face).replace('.', '_').split('_')[0]
            face_names.append(name)
            prewhitened = detect_face.prewhiten(img)
            face_list.append(prewhitened)

        if len(face_list) < 1:
            return face_names, None

        face_images = np.stack(face_list)  # 人脸图像容器
        num_users = face_images.shape[0]  # 用户数量
        face_embeddings = np.zeros((num_users, 512))  # 人脸特征容器

        """将人脸图像编码为512维特征向量"""
        for i in range(num_users // 32 + 1):
            feed_dict = {self.net_inputs: face_images[i * 32:(i + 1) * 32, :, :, :], self.phase_train: False}
            face_embeddings[i * 32:(i + 1) * 32, :] = self.sess.run(self.net_outputs, feed_dict=feed_dict)

        del face_list
        del face_images
        gc.collect()
        return face_names, face_embeddings

    def load_and_align_data(self, img):
        """检测图像中的人脸并裁剪"""
        height, width = img.shape[:2]

        bboxes, _ = detect_face.detect_face(img=img,
                                            minsize=20 * self.face_size,
                                            pnet=self.pnet, rnet=self.rnet, onet=self.onet,
                                            threshold=[0.6, 0.7, 0.7],
                                            factor=0.709)
        if len(bboxes) < 1:
            return False, np.zeros((0, 0)), []

        bboxes[:, 0] = np.maximum(bboxes[:, 0] - 20, 0)
        bboxes[:, 1] = np.maximum(bboxes[:, 1] - 20, 0)
        bboxes[:, 2] = np.minimum(bboxes[:, 2] + 20, width - 1)
        bboxes[:, 3] = np.minimum(bboxes[:, 3] + 20, height - 1)
        bboxes = bboxes.astype(int)

        """裁剪标准尺寸的人脸图像并做预白化处理"""
        cropped_list = []
        for i in range(len(bboxes)):
            cropped = img[bboxes[i, 1]:bboxes[i, 3], bboxes[i, 0]:bboxes[i, 2], :]
            aligned = cv2.resize(cropped, (160, 160))
            if i == 0:
                self.face_buffer = aligned
            prewhitened = detect_face.prewhiten(aligned)
            cropped_list.append(prewhitened)
        cropped_images = np.stack(cropped_list)

        return True, bboxes, cropped_images

    def camera_switch(self):
        """摄像头开关"""
        if not self.timer.isActive():
            flag = self.cap.open(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not flag:
                QtWidgets.QMessageBox.warning(self, '', '无法启动摄像头', buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer.start(30)  # 每30ms捕获一帧图像
                self.button_switch.setText('关闭')
        else:
            self.timer.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_switch.setText('启动')

    def sign_in(self):
        """签到"""
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, '警告', '摄像头未启动', buttons=QtWidgets.QMessageBox.Ok)
            return
        if self.face_buffer is None:
            return
        face = self.face_buffer.copy()
        show = QtGui.QImage(face.data, face.shape[1], face.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_face.setPixmap(QtGui.QPixmap.fromImage(show))
        prewhitened = detect_face.prewhiten(face)
        feed_dict = {self.net_inputs: np.stack([prewhitened]), self.phase_train: False}
        emb = self.sess.run(self.net_outputs, feed_dict=feed_dict)
        if self.known_embeddings is None:
            QtWidgets.QMessageBox.information(self, '签到失败', '用户未注册', buttons=QtWidgets.QMessageBox.Ok)
        else:
            face_distances = np.sqrt(np.sum(np.square(emb[0, :] - self.known_embeddings[:, :]), axis=1))
            min_distance = np.min(face_distances)
            if min_distance < self.tolerance:
                name = self.known_names[int(np.argmin(face_distances))]
                self.show_record.setText(self.show_record.toPlainText() + time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time())) + name + '已签到！\n')
                self.record.write(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time())) + name + '已签到！\n')
                self.record.close()
                self.record = open('签到记录.txt', 'a')
                QtWidgets.QMessageBox.information(self, '签到成功', time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime(time.time())) + name + '已签到！',
                                                  buttons=QtWidgets.QMessageBox.Ok)
            else:
                QtWidgets.QMessageBox.information(self, '签到失败', '用户未注册', buttons=QtWidgets.QMessageBox.Ok)
        self.label_show_face.setPixmap(QtGui.QPixmap.fromImage(self.face_background))

    def register_user(self):
        """注册当前画面中的人脸"""
        if not self.admin_mode:
            QtWidgets.QMessageBox.warning(self, '警告', '未进入管理模式', buttons=QtWidgets.QMessageBox.Ok)
            return
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, '警告', '摄像头未启动', buttons=QtWidgets.QMessageBox.Ok)
            return
        if self.face_buffer is None:
            return
        face = self.face_buffer.copy()
        show = QtGui.QImage(face.data, face.shape[1], face.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_face.setPixmap(QtGui.QPixmap.fromImage(show))
        name, ok = QtWidgets.QInputDialog.getText(self, '用户注册', '请输入用户姓名', QtWidgets.QLineEdit.Normal, '')
        if ok:
            prewhitened = detect_face.prewhiten(face)
            feed_dict = {self.net_inputs: np.stack([prewhitened]), self.phase_train: False}
            emb = self.sess.run(self.net_outputs, feed_dict=feed_dict)
            if self.known_embeddings is None:
                self.known_embeddings = emb.copy()
            else:
                self.known_embeddings = np.concatenate((self.known_embeddings, emb), axis=0)
            self.known_names.append(name)
            imageio.imsave(os.path.join(self.database_dir, name + '.jpg'), face)
            QtWidgets.QMessageBox.information(self, '注册成功', name + '已注册！', buttons=QtWidgets.QMessageBox.Ok)
        self.label_show_face.setPixmap(QtGui.QPixmap.fromImage(self.face_background))

    def quit(self):
        """关闭软件"""
        if self.timer.isActive():
            self.timer.stop()
            self.cap.release()
        self.close()

    def admin_switch(self):
        """管理模式开关"""
        if self.admin_mode:
            text, ok = QtWidgets.QInputDialog.getText(self, '管理模式开关', '输入密码以退出管理模式', QtWidgets.QLineEdit.Normal, '')
            if ok:
                if text == self.admin_password:
                    self.admin_mode = False
                else:
                    QtWidgets.QMessageBox.warning(self, '警告', '密码错误', buttons=QtWidgets.QMessageBox.Ok)
                    self.admin_switch()
        else:
            text, ok = QtWidgets.QInputDialog.getText(self, '管理模式开关', '输入密码以进入管理模式', QtWidgets.QLineEdit.Normal, '')
            if ok:
                if text == self.admin_password:
                    self.admin_mode = True
                else:
                    QtWidgets.QMessageBox.warning(self, '警告', '密码错误', buttons=QtWidgets.QMessageBox.Ok)
                    self.admin_switch()

    def change_password(self):
        """修改密码"""
        if not self.admin_mode:
            QtWidgets.QMessageBox.warning(self, '警告', '未进入管理模式', buttons=QtWidgets.QMessageBox.Ok)
            return
        text, ok = QtWidgets.QInputDialog.getText(self, '修改密码', '请输入新密码', QtWidgets.QLineEdit.Normal, '')
        if ok:
            self.admin_password = text

    def import_user(self):
        """导入用户"""
        if not self.admin_mode:
            QtWidgets.QMessageBox.warning(self, '警告', '未进入管理模式', buttons=QtWidgets.QMessageBox.Ok)
            return
        image_list = QtWidgets.QFileDialog.getOpenFileNames(None, '请选择图片', 'D:\\',
                                                            '所有图片 (*.jpg;*.jpeg;*.jpe;*.jfif;*.png;*.bmp;*.dib;*.tif;*.tiff)')
        for image in image_list[0]:
            name = os.path.basename(image).split('.')[0]
            img = imageio.imread(image)
            if img.shape[2] > 3:
                img = img[..., :3]
            face_flag, bboxes, cropped_images = self.load_and_align_data(img)
            if len(cropped_images) == 1:
                feed_dict = {self.net_inputs: cropped_images, self.phase_train: False}
                emb = self.sess.run(self.net_outputs, feed_dict=feed_dict)
                if self.known_embeddings is None:
                    self.known_embeddings = emb.copy()
                else:
                    self.known_embeddings = np.concatenate((self.known_embeddings, emb), axis=0)
                self.known_names.append(name)
                imageio.imsave(os.path.join(self.database_dir, name + '.jpg'), cropped_images[0])

    def delete_user(self):
        """删除用户"""
        if not self.admin_mode:
            QtWidgets.QMessageBox.warning(self, '警告', '未进入管理模式', buttons=QtWidgets.QMessageBox.Ok)
            return
        if self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, '警告', '摄像头未关闭', buttons=QtWidgets.QMessageBox.Ok)
            return
        image_list = QtWidgets.QFileDialog.getOpenFileNames(None, '请选择图片', self.database_dir, '所有图片 (*.jpg;)')
        for image in image_list[0]:
            if os.path.dirname(image) == os.path.abspath(self.database_dir).replace('\\', '/'):
                os.remove(image)
        self.known_names, self.known_embeddings = self.load_faces()

    def show_camera(self):
        """执行人脸检测识别并显示摄像头画面"""
        _, frame = self.cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=1 / self.scale_factor, fy=1 / self.scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        """人脸检测"""
        face_flag, bboxes, cropped_images = self.load_and_align_data(rgb_small_frame)

        if not face_flag:
            self.face_buffer = None
        else:
            """人脸识别"""
            feed_dict = {self.net_inputs: cropped_images, self.phase_train: False}
            emb = self.sess.run(self.net_outputs, feed_dict=feed_dict)
            face_names = []

            for i in range(len(emb)):
                if self.known_embeddings is None:
                    face_names.append('未注册用户')
                else:
                    face_distances = np.sqrt(np.sum(np.square(emb[i, :] - self.known_embeddings[:, :]), axis=1))
                    min_distance = np.min(face_distances)
                    if min_distance < self.tolerance:
                        face_names.append(self.known_names[int(np.argmin(face_distances))] + ' '
                                          + str(round(min((1 - min_distance) * 400 / 3, 100), 2)) + '%')
                    else:
                        face_names.append('未注册用户')

            """绘制人脸定位框及姓名"""
            for face in range(len(bboxes)):
                top = bboxes[face, 1] * self.scale_factor
                right = bboxes[face, 2] * self.scale_factor
                bottom = bboxes[face, 3] * self.scale_factor
                left = bboxes[face, 0] * self.scale_factor
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
                cv2.rectangle(frame, (left, bottom), (right, bottom + (bottom - top) // 8), (0, 0, 255), cv2.FILLED)

                name = face_names[face]
                pilimg = Image.fromarray(frame)
                draw = ImageDraw.Draw(pilimg)
                font = ImageFont.truetype("./resources/simhei.ttf", (bottom - top) // 8)
                draw.text((left, bottom), name, (255, 255, 255), font=font)
                frame = np.array(pilimg)

        """将图像转为QImage对象并显示"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        show = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(show))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
