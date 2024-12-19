# Hello World Projesi

Bu proje, terminalde ekrana "Hello, World!" yazdıran basit bir C++ uygulamasıdır.

## Kurulum

Projenin derlenmesi için **g++** veya benzeri bir C++ derleyicisine ihtiyacınız vardır.
- Burada aynı zamanda Bizim vscode ayarlarımızda mevcut eğer bir #include <iostream> demişşek
ve altı kırmızı görünüyorsa burada VsCode içerisinde /usr/include/gcc yi configlerde dememiz gerekiyor.

### Linux/MacOS

1. Proje dizinine gidin ve derleyin:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make
   ```

2. Uygulamayı çalıştırın:
   ```bash
   ./HelloWorld
   ```

### Windows

1. Proje dizinine gidin ve derleyin:
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

2. Uygulamayı çalıştırın:
   ```bash
   .\Debug\HelloWorld.exe
   ```

## Katkıda Bulunma

Katkıda bulunmak isterseniz, lütfen bir pull request gönderin veya bir issue açın.

## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına bakın.

