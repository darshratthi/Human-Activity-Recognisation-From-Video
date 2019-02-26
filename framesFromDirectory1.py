import cv2
import os

def frames(file,name_of_video_file):
	vidcap=cv2.VideoCapture(file)
	success,image=vidcap.read()
	count=0
	success=True
	count1=1
	while success:
		name='./humanActivityRecognisationImages/train_dir/'+name_of_video_file+'/'+name_of_video_file+'.'+str(count1)+'.jpg'
		if count%3==0:
			cv2.imwrite(name,image)#save frame as JPEG file
			count1+=1

		success,image=vidcap.read()
		print('Read a new frame',success)
		count+= 1

	vidcap.release()
	cv2.destroyAllWindows()


for file in os.listdir("./humanActivityRecognisationVideos"):
	if file.endswith(".avi"):
		name_of_video_file=os.path.splitext(file)[0]
		try:
			if not os.path.exists('humanActivityRecognisationImages/train_dir/'+name_of_video_file):
				os.makedirs('humanActivityRecognisationImages/train_dir/'+name_of_video_file)
		except OSError:
			print('ERROR: Creating directory')

		path=os.path.join("humanActivityRecognisationVideos",file)
		print(path)
		frames(path,name_of_video_file)


