#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_predictor = new networkPredictor();

    m_predictor->loadNetwork();
    m_predictor->predict();

}

MainWindow::~MainWindow()
{
    delete ui;
}

