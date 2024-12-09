import React, { useState } from 'react';
import { View, Button, Text, ActivityIndicator, Modal, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';

const UploadScreen = ({ navigation }) => {
  const [video, setVideo] = useState(null);
  const [faceImage, setFaceImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [modalType, setModalType] = useState(null);

  const pickVideoFromFileSystem = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({ type: 'video/*' });
      if (result.type === 'success') {
        setVideo({ uri: result.uri, name: result.name, type: result.mimeType });
      }
    } catch (error) {
      console.log(error);
      Alert.alert('Error', 'Failed to pick a video from the file system.');
    }
  };

  const pickVideoFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: true,
        quality: 1,
      });
      if (!result.cancelled) {
        const { uri, type } = result.assets[0];
        setVideo({ uri, name: uri.split('/').pop(), type });
      }
    } catch (error) {
      console.log(error);
      Alert.alert('Error', 'Failed to pick a video from the gallery.');
    }
  };

  const pickImageFromFileSystem = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({ type: 'image/*' });
      if (result.type === 'success') {
        setFaceImage({ uri: result.uri, name: result.name, type: result.mimeType });
      }
    } catch (error) {
      console.log(error);
      Alert.alert('Error', 'Failed to pick an image from the file system.');
    }
  };

  const pickImageFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });
      if (!result.cancelled && result.assets && result.assets.length > 0) {
        const { uri, type } = result.assets[0];
        setFaceImage({ uri, name: uri.split('/').pop(), type });
      }
    } catch (error) {
      console.log(error);
      Alert.alert('Error', 'Failed to pick an image from the gallery.');
    }
  };

  const startTrimming = async () => {
    setLoading(true);
    const formData = new FormData();
    formData.append('video', {
      uri: video.uri,
      type: video.mimeType || 'video/mp4',
      name: video.name || 'video.mp4',
    });
    formData.append('face', {
      uri: faceImage.uri,
      type: faceImage.mimeType || 'image/jpeg',
      name: faceImage.name || 'face_image.jpg',
    });
  
    try {
      const response = await axios.post('http://http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      navigation.navigate('TrimResults', { trimmedVideoUrl: response.data.trimmedVideoUrl });
    } catch (error) {
      console.log(error);
    } finally {
      setLoading(false);
    }
  };
  

  const openModal = (type) => {
    setModalType(type);
    setModalVisible(true);
  };

  const handlePickFromFileSystem = () => {
    setModalVisible(false);
    if (modalType === 'video') {
      pickVideoFromFileSystem();
    } else {
      pickImageFromFileSystem();
    }
  };

  const handlePickFromGallery = () => {
    setModalVisible(false);
    if (modalType === 'video') {
      pickVideoFromGallery();
    } else {
      pickImageFromGallery();
    }
  };

  return (
    <View style={{ padding: 20 }}>
      <Button title="Upload Video" onPress={() => openModal('video')} />
      <Text>{video ? (video.name || 'Video selected from gallery') : 'No video selected'}</Text>
      <Button title="Upload Face" onPress={() => openModal('face')} />
      <Text>{faceImage ? (faceImage.name || 'Image selected from gallery') : 'No face image selected'}</Text>
      <Button title="Start Trimming" onPress={startTrimming} disabled={loading || !video || !faceImage} />
      {loading && <ActivityIndicator size="large" color="#0000ff" />}
      
      <Modal
        transparent={true}
        visible={modalVisible}
        animationType="slide"
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalView}>
            <Text style={styles.modalTitle}>Select Source</Text>
            <TouchableOpacity style={styles.button} onPress={handlePickFromFileSystem}>
              <Text style={styles.buttonText}>File System</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={handlePickFromGallery}>
              <Text style={styles.buttonText}>Gallery</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={() => setModalVisible(false)}>
              <Text style={styles.buttonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalView: {
    width: 300,
    padding: 20,
    backgroundColor: 'white',
    borderRadius: 10,
    alignItems: 'center',
  },
  modalTitle: {
    fontSize: 18,
    marginBottom: 20,
  },
  button: {
    padding: 10,
    marginVertical: 5,
    backgroundColor: '#007bff',
    borderRadius: 5,
    width: '80%',
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
  },
});

export default UploadScreen;
