import 'dart:io';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as imageLib;
import 'package:object_detection/tflite/classifier.dart';
import 'package:object_detection/utils/image_utils.dart';
import 'package:tflite_flutter/tflite_flutter.dart';


/**
 * CETTE CLASSE ABRITE LES METHODES QUI SONT UTILISEES POUR RECEVOIR LES RESULTATS
 * DU MODEL DE MANIERE ASYNCHRONE, POUR NE PAS RALENTIR L'UI DE L'APP
 */

/// C'est quoi un [Isolate] ?
///   On the surface an Isolate might just seem like another way of executing a task
///   in an [asynchronous] manner (see my Dart tutorial on async here for more on that)
///   but there is a key difference. Async operations in Dart operate on the same thread
///   as the user interface whereas an [Isolate] gets its own [thread].  Therefore, if
///   you want to execute a process that is fairly intense and you want to make sure
///   you keep your app responsive and snappy in the eyes of the user, then you should
///   consider using Isolates.
///   Lien de l'explication : https://codingwithjoe.com/dart-fundamentals-isolates/


/// Manages separate Isolate instance for inference
class IsolateUtils {
  static const String DEBUG_NAME = "InferenceIsolate";

  Isolate _isolate;
  ReceivePort _receivePort = ReceivePort();
  SendPort _sendPort;

  SendPort get sendPort => _sendPort;

  void start() async {
    _isolate = await Isolate.spawn<SendPort>(
      entryPoint,
      _receivePort.sendPort,
      debugName: DEBUG_NAME,
    );

    _sendPort = await _receivePort.first;
  }

  static void entryPoint(SendPort sendPort) async {
    final port = ReceivePort();
    sendPort.send(port.sendPort);

    await for (final IsolateData isolateData in port) {
      if (isolateData != null) {
        Classifier classifier = Classifier(
            interpreter:
                Interpreter.fromAddress(isolateData.interpreterAddress),
            labels: isolateData.labels);
        imageLib.Image image =
            ImageUtils.convertCameraImage(isolateData.cameraImage);
        if (Platform.isAndroid) {
          image = imageLib.copyRotate(image, 90);
        }
        Map<String, dynamic> results = classifier.predict(image);
        isolateData.responsePort.send(results);
      }
    }
  }
}

/// Bundles data to pass between Isolate
class IsolateData {
  CameraImage cameraImage;
  int interpreterAddress;
  List<String> labels;
  SendPort responsePort;

  IsolateData(
    this.cameraImage,
    this.interpreterAddress,
    this.labels,
  );
}
