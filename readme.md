# my_own_cnn

Учебный каркас слоев нейросетей с реализациями для CPU и CUDA (NHWC).

## Сборка
```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## Реализованные слои
- Pooling: max, average, median, min (forward/backward/grad CPU, заглушки CUDA)
- BatchNorm: forward/backward CPU и CUDA-заглушки
- Softmax: forward/backward CPU и CUDA-заглушки
- Cross entropy with logits: прямой и обратный проход
- Convolution: CPU, CUDA-заглушка и cuDNN-обертка

Все операции используют порядок данных NHWC. Простые CPU версии предназначены для надежных тестов в GoogleTest.
