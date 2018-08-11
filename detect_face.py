from random import randint
import cv2
import os
import glob
      
# CASCADE="Face_cascade.xml"
FRONT_CASCADE = 'haarcascade_frontalface_default.xml'
FRONT_FACE_CASCADE=cv2.CascadeClassifier(FRONT_CASCADE)

PROFILE_CASCADE = 'haarcascade_profileface.xml'
PROFILE_FACE_CASCADE=cv2.CascadeClassifier(PROFILE_CASCADE)

def detect_front_faces(image_path):

	image=cv2.imread(image_path)
	image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	height, width = image.shape[:2]
	# faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
	faces = FRONT_FACE_CASCADE.detectMultiScale(image_grey, scaleFactor = 1.16, minNeighbors = 5, minSize = (int(height / 5), int(width / 7)))

	for x,y,w,h in faces:
		sub_img=image[y-10:y+h+10,x-10:x+w+10]
		print(image_path)
		extract_path = '/'.join(image_path.split('/')[:-1]) + '/extracted'
		# print(extract_path)
		if not os.path.exists(extract_path):
			os.mkdir(extract_path)
		write_path = extract_path + '/' + image_path.split('.')[0][-6:] + '_front.jpg'
		# print(write_path)
		cv2.imwrite(write_path, sub_img)
		# os.chdir("../")
		# cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

	# cv2.imshow("Faces Found",image)
	# if (cv2.waitKey(0) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
	# 	cv2.destroyAllWindows()


def detect_profile_faces(image_path):

	image=cv2.imread(image_path)
	image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	height, width = image.shape[:2]
	# faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
	faces = PROFILE_FACE_CASCADE.detectMultiScale(image_grey, scaleFactor = 1.16, minNeighbors = 5, minSize = (int(height / 7), int(width / 9)))

	for x,y,w,h in faces:
		if h / w > 3:
			continue
		sub_img=image[y-10:y+h+10,x-10:x+w+10]
		print(image_path)
		extract_path = '/'.join(image_path.split('/')[:-1]) + '/extracted'
		# print(extract_path)
		if not os.path.exists(extract_path):
			os.mkdir(extract_path)
		write_path = extract_path + '/' + image_path.split('.')[0][-6:] + '_profile.jpg'
		# print(write_path)
		cv2.imwrite(write_path, sub_img)
		# os.chdir("../")
		# cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

	# cv2.imshow("Faces Found",image)
	# if (cv2.waitKey(0) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
	# 	cv2.destroyAllWindows()

if __name__ == "__main__":
	basepath = "C:/Users/73175/Videos/IQIYI_VID_DATA_Part1/IMAGE_TRAIN/"
	imgs = glob.glob(basepath + '*/*.jpg')
	print(len(imgs))
	for img in imgs:
		img = img.replace('\\', '/')
		detect_front_faces(img)
		detect_profile_faces(img)