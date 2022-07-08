#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    m_scene = new AnnotationScene();
    ui->graphicsView->setScene(m_scene);
    ui->graphicsView->show();

}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{
    QString dir_path = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                         "",
                                                         QFileDialog::ShowDirsOnly
                                                         | QFileDialog::DontResolveSymlinks);

    QDir dir(dir_path);

    QStringList images = dir.entryList(QStringList() << "*.jpg" << "*.JPG",QDir::Files);
    foreach(QString filename, images) {
        QString img_path = dir.filePath(filename);
        cv::Mat mat = cv::imread(img_path.toStdString());
        m_images.push_back(mat);
    }

}

