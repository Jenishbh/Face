import React, { useState, useEffect } from 'react';
import { View, Button, Text, Alert, StyleSheet, Dimensions } from 'react-native';
import { Video } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';

const TrimResultsScreen = ({ route }) => {
  const { trimmedVideoUrl } = route.params;
  const [videoUri, setVideoUri] = useState(trimmedVideoUrl);

  const saveVideo = async () => {
    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission required', 'Permission to access media library is required to save videos.');
        return;
      }

      const fileUri = FileSystem.documentDirectory + 'trimmed_video.mp4';
      const result = await FileSystem.downloadAsync(trimmedVideoUrl, fileUri);

      const asset = await MediaLibrary.createAssetAsync(result.uri);
      await MediaLibrary.createAlbumAsync('Download', asset, false);

      Alert.alert('Video saved successfully', `Saved to gallery as ${asset.filename}`);
    } catch (error) {
      console.log('Error saving video:', error);
      Alert.alert('Failed to save video');
    }
  };

  return (
    <View style={styles.container}>
      <Text>Trimmed Video URL: {trimmedVideoUrl}</Text>
      <Video
        source={{ uri: videoUri }}
        rate={1.0}
        volume={1.0}
        isMuted={false}
        resizeMode="cover"
        shouldPlay
        isLooping
        style={styles.video}
        useNativeControls
      />
      <Button title="Save Video" onPress={saveVideo} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  video: {
    width: Dimensions.get('window').width,
    height: 200,
    backgroundColor: 'black',
  },
});

export default TrimResultsScreen;
