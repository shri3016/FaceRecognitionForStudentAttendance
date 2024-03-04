import React, { useState } from "react";
import { StyleSheet, Text, View, TouchableOpacity } from "react-native";
import { useRoute } from "@react-navigation/native";

function SubjectDetails() {
  const route = useRoute();
  const { id } = route.params;

  const [result, setResult] = useState(null);
  const [cattendanceresult, setCattendanceResult] = useState(null);
  const [mattendanceresult, setMattendanceResult] = useState(null);
  const [showCreateButton, setShowCreateButton] = useState(false);

  const runFaceDetection = async () => {
    try {
      const response = await fetch("http://192.168.72.6:5000/teachers-detecting-faces", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });

      if (response.ok) {
        const data = await response.json();
        setResult(data.message);
        setShowCreateButton(true);
      } else {
        setResult("Error: " + response.status);
      }
    } catch (error) {
      console.log(error);
      setResult("Error: " + error.message);
    }
  };

  const createAttendance = async () => {
    try {
      const formData = new FormData();
      formData.append("subject", id);

      const response = await fetch("http://192.168.72.6:5000/creating-attendance-file", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log(data.message);
        setCattendanceResult(data.message);
      } else {
        console.log("Error: " + response.status);
      }
    } catch (error) {
      console.log(error);
    }
  };

  const markAttendance = async () => {
    try {
      const formData = new FormData();
      formData.append("subject", id);

      const response = await fetch("http://192.168.72.6:5000/marking-attendance", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log(data.message);
        setMattendanceResult(data.message);
      } else {
        console.log("Error: " + response.status);
      }
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.subjectName}>{id}</Text>
      <TouchableOpacity style={styles.button} onPress={runFaceDetection}>
        <Text style={styles.buttonText}>Run Face Detection</Text>
      </TouchableOpacity>
      {result && <Text style={styles.resultText}>{result}</Text>}
      {showCreateButton && (
        <TouchableOpacity style={[styles.button, styles.createButton]} onPress={createAttendance}>
          <Text style={styles.buttonText}>Create Attendance</Text>
        </TouchableOpacity>
      )}
        {cattendanceresult && <Text style={styles.resultText}>{cattendanceresult}</Text>}

      {showCreateButton && (
        <TouchableOpacity style={[styles.button, styles.markButton]} onPress={markAttendance}>
          <Text style={styles.buttonText}>Mark Attendance</Text>
        </TouchableOpacity>
      )}
      {mattendanceresult && <Text style={styles.resultText}>{mattendanceresult}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  subjectName: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 20,
  },
  button: {
    backgroundColor: "#344B79",
    padding: 10,
    borderRadius: 8,
    marginBottom: 20,
  },
  createButton: {
    backgroundColor: "#344B79", // Set the same color as the run button
  },
  markButton: {
    backgroundColor: "#344B79", // Set the same color as the run button
  },
  buttonText: {
    color: "white",
    fontSize: 16,
  },
  resultText: {
    fontSize: 16,
  },
});

export default SubjectDetails;
