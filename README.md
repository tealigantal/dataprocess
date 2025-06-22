Hi everyone, 不洋蛋了，直奔主题
我这个程序主要包含了：python文件（执行程序），topics（雅思口语的topic），一个json文件（咱们自己的转录和打分结合的文本文件）。
程序主要的运行步骤：
                  1、将json文件喂给deepseek，让deepseek参考并生成一段类似的组（input是转录的文本，output是输出的打分）
                  2、由人来判断打分是否合理，输入y/n/进行数据复制，y表示保留数据，n表示不保留，进行数据复制表示数据的生成没有问题，可以直接开始数据的复制。
程序所需要的环境：pip install openai python-dotenv
                在系统环境变量中加入DEEPSEEK_API_KEY 这个变量（具体的值我会单独私发，因为这个仓库是公开的所以会由盗用风险）
                
