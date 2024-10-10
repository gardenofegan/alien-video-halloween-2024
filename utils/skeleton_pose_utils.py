import cv2
import numpy as np

class SkeletonDrawer:
	def __init__(self):
		self.read_skeleton_images()

	def read_skeleton_images(self):
		# Load bone image from a local file
		self.bone_image = cv2.imread("path/to/your/local/bone.png", cv2.IMREAD_UNCHANGED)
		self.foot_image = cv2.imread("images/foot.png", cv2.IMREAD_UNCHANGED)
		self.hand_image = cv2.imread("images/hand.png", cv2.IMREAD_UNCHANGED)
		self.torso_image = cv2.imread("images/torso.png", cv2.IMREAD_UNCHANGED)
		self.skull_image = cv2.imread("images/skull.png", cv2.IMREAD_UNCHANGED)

	def draw_skeleton(self, image, keypoints):
		for part in ['skull', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']:
			self.draw_body_part(image, keypoints, part)
		return image

	def draw_body_part(self, image, keypoints, part):
		if part == 'skull':
			self.draw_skull(image, keypoints[0:5])
		elif part == 'torso':
			self.draw_torso(image, keypoints[[5, 6, 11, 12]])
		elif part == 'left_arm':
			self.draw_bone(image, keypoints[[5, 7]])
			self.draw_bone(image, keypoints[[7, 9]])
		elif part == 'right_arm':
			self.draw_bone(image, keypoints[[6, 8]])
			self.draw_bone(image, keypoints[[8, 10]])
		elif part == 'left_leg':
			self.draw_bone(image, keypoints[[11, 13]])
			self.draw_bone(image, keypoints[[13, 15]])
		elif part == 'right_leg':
			self.draw_bone(image, keypoints[[12, 14]])
			self.draw_bone(image, keypoints[[14, 16]])

	def draw_skull(self, image, keypoints):
		src_pts = np.float32([[0, 0], [self.skull_image.shape[1], 0], [self.skull_image.shape[1]//2, self.skull_image.shape[0]]])
		dst_pts = np.float32([keypoints[0], keypoints[1], (keypoints[3] + keypoints[4]) / 2])
		M = cv2.getAffineTransform(src_pts, dst_pts)
		rows, cols = image.shape[:2]
		skull = cv2.warpAffine(self.skull_image, M, (cols, rows))
		mask = skull[:,:,3] / 255.0
		image[:] = (1.0 - mask[:,:,np.newaxis]) * image + mask[:,:,np.newaxis] * skull[:,:,:3]

	def draw_torso(self, image, keypoints):
		src_pts = np.float32([[0, 0], [self.torso_image.shape[1], 0], [self.torso_image.shape[1]//2, self.torso_image.shape[0]]])
		dst_pts = np.float32([keypoints[0], keypoints[1], (keypoints[2] + keypoints[3]) / 2])
		M = cv2.getAffineTransform(src_pts, dst_pts)
		rows, cols = image.shape[:2]
		torso = cv2.warpAffine(self.torso_image, M, (cols, rows))
		mask = torso[:,:,3] / 255.0
		image[:] = (1.0 - mask[:,:,np.newaxis]) * image + mask[:,:,np.newaxis] * torso[:,:,:3]

	def draw_bone(self, image, keypoints):
		angle = np.arctan2(keypoints[1][1] - keypoints[0][1], keypoints[1][0] - keypoints[0][0])
		length = np.linalg.norm(keypoints[1] - keypoints[0])
		bone = cv2.resize(self.bone_image, (int(length), self.bone_image.shape[0]))
		M = cv2.getRotationMatrix2D((0, bone.shape[0]//2), angle * 180 / np.pi, 1)
		M[:, 2] += keypoints[0] - [0, bone.shape[0]//2]
		rows, cols = image.shape[:2]
		bone = cv2.warpAffine(bone, M, (cols, rows))
		mask = bone[:,:,3] / 255.0
		image[:] = (1.0 - mask[:,:,np.newaxis]) * image + mask[:,:,np.newaxis] * bone[:,:,:3]