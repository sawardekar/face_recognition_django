from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from recognition.camera import FaceDetect
# Create your views here.
def index(request):
	return render(request, 'recognition/home.html')

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
		
def facecam_feed(request):
	return StreamingHttpResponse(gen(FaceDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame')