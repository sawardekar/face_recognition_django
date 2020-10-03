from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle,os
from django.conf import settings

def model_train():
	# load the face embeddings
	print("[INFO] loading face embeddings...")
	embeddings = os.path.sep.join([settings.BASE_DIR, "output\\embeddings.pickle"])
	data = pickle.loads(open(embeddings, "rb").read())


	# encode the labels
	print("[INFO] encoding labels...")
	le = LabelEncoder()
	labels = le.fit_transform(data["names"])

	# train the model used to accept the 128-d embeddings of the face and
	# then produce the actual face recognition
	print("[INFO] training model...")
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(data["embeddings"], labels)

	# write the actual face recognition model to disk
	recognizers = os.path.sep.join([settings.BASE_DIR, "output\\recognizer.pickle"])
	f = open(recognizers, "wb")
	f.write(pickle.dumps(recognizer))
	f.close()

	# write the label encoder to disk
	le_pickle = os.path.sep.join([settings.BASE_DIR, "output\\le.pickle"])
	f = open(le_pickle, "wb")
	f.write(pickle.dumps(le))
	f.close()