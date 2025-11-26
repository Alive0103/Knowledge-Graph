@echo off
chcp 65001 >nul
echo ========================================
echo Elasticsearch 数据导入流程
echo ========================================
echo.

echo [步骤 1/3] 测试 ES 连接有效性...
python testes\test_es_aliyun.py
if %errorlevel% neq 0 (
    echo 连接测试失败，请检查配置后重试
    pause
    exit /b 1
)
echo.

echo [步骤 2/3] 创建索引...
python createES.py
if %errorlevel% neq 0 (
    echo 索引创建失败，请检查错误信息
    pause
    exit /b 1
)
echo.

echo [步骤 3/3] 导入数据...
python toesdata.py
if %errorlevel% neq 0 (
    echo 数据导入失败，请检查错误信息
    pause
    exit /b 1
)
echo.

echo ========================================
echo 所有步骤完成！
echo ========================================
pause

