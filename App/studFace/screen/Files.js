import React, { useEffect, useState } from "react";
import { Text, FlatList, StyleSheet, View } from "react-native";
import FilesList from "../components/FilesList";
import AsyncStorage from '@react-native-async-storage/async-storage';

function Files() {
  const [user, setUser] = useState(null);
  const [subjects, setSubjects] = useState([]);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = await AsyncStorage.getItem('token');

        const response = await fetch("http://192.168.72.6:3000/api/v1/user", {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        const data = await response.json();

        if (response.ok) {
          setUser(data.firstName + ' ' + data.lastName);
          setSubjects(data.subjects);
        } else {
          console.error(data.message);
        }
      } catch (error) {
        console.error(error);
      }
    };

    fetchUserData();
  }, []);

  return (
    <View style={styles.container}>
      <View style={styles.container2}>
        <Text style={styles.subjectsTitle}>Files</Text>
        <FlatList
          data={subjects}
          renderItem={({ item }) => (
            <FilesList text={item} id={item} />
          )}
          keyExtractor={(item, index) => item}
          style={{ flex: 1 }}
        />
      </View>
    </View>
  );
}

export default Files;

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  container2: {
    flex: 1,
    alignItems: "center",
  },
  subjectsTitle: {
    fontWeight: "bold",
    fontSize: 24,
    backgroundColor: '#162544',
    width: "100%",
    padding: 20,
    color: 'white',
    alignItems: "center"
  },
});
