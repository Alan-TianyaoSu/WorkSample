# 词袋配置
word_bag = {
            '文书类型': ['民事裁定', '民事判决', '执行裁定', '刑事判决', '结案通知',
                        '行政裁定', '刑事裁定', '执行通知', '受理案件通知', '执行决定',
                        '支付令', '行政判决', '民事调解', '应诉通知', '其他',

                        '民事裁定书', '民事判决书', '执行裁定书', '刑事判决书', '结案通知书',
                        '行政裁定书', '刑事裁定书', '执行通知书', '受理案件通知书', '执行决定书',
                        '支付令', '行政判决书', '民事调解书', '应诉通知书', '其他'],

            '原告': ['诉称，', '诉称：', '提出诉讼请求'],
            '被告': ['辩称，', '辩称：', '未作答辩'],
            '法院': ['经审查', '经审理查明'],
            '诉称': ['诉称，', '诉称：', '提出诉讼请求', '辩称，', '辩称：', '未作答辩', '上诉请求','审查认为'],
            '头部': ['本院','审判员','诉称','保全申请']

        }

# 词表映射
word_mapping = {
    '民事裁定': '民事裁定书','民事判决': '民事判决书','执行裁定': '执行裁定书',
    '刑事判决': '刑事判决书','结案通知': '结案通知书','行政裁定': '行政裁定书',
    '刑事裁定': '刑事裁定书','执行通知': '执行通知书','受理案件通知': '受理案件通知书',
    '执行决定': '执行决定书','支付令': '支付令','行政判决': '行政判决书',
    '民事调解': '民事调解书','应诉通知': '应诉通知书','其他': '其他'
}

Judge_Tokenizer = ['代理', '审判', '书记', '代书记', '人民陪审员',
                   '一', '二', '三', '四', '五', '六', '七', '八', '九', '零', '贰']

Relevant_Person = [
    '原告：', '被告：', '申请执行人：', '被执行人：', '异议人（申请执行人）：', '委托诉讼代理人：', '法定代表人：', '第三人：', '上诉人（原审被告）：',
    '被上诉人（原审原告）：', '上诉人（原审原告）：', '被上诉人（原审被告）：', '经营者：', '法人代表人：', '申请人：', '被申请人：', '原告人：', '被告人：',
    '法定代理人：', '委托诉讼代理人：', '上诉人（一审起诉人）：', '原审被告：', '原审原告：', '原审第三人：','一审被告','一审原告','一审第三人', '再审申请人','再审被申请人',
    '再审原审被告','再审原审原告','再审第三人','再审上诉人','再审被上诉人','再审原审第三人','再审第三人','再审法定代表人','再审委托代理人','再审法定代表人','再审委托代理人',
    '二审原告', '二审被告', '二审第三人', '二审法定代表人', '二审委托代理人', '二审法定代表人', '二审委托代理人', '二审上诉人', '二审被上诉人', '二审原审被告', '二审原审原告', '二审原审第三人',

]


Deffendant_Split = ['提交书面意见称','辩称','提交意见称','未作答辩','未作辩称','未提出答辩','未提出辩称']


Classification_mapping = {
    '缔约过失合同纠纷': '合同纠纷', '确认合同有效纠纷': '合同纠纷', '确认合同无效纠纷': '合同纠纷',
    '债权人代位权纠纷': '合同纠纷', '债权人撤销权纠纷': '合同纠纷', '债权转让合同纠纷': '合同纠纷', 
    '债务转移合同纠纷': '合同纠纷', '债权债务概括转移合同纠纷': '合同纠纷', '悬赏广告纠纷': '合同纠纷',
    '买卖合同纠纷': '合同纠纷', '分期付款买卖合同纠纷': '合同纠纷','凭样品买卖合同纠纷': '合同纠纷',
    '试用买卖合同纠纷': '合同纠纷', '互易纠纷': '合同纠纷', '国际货物买卖合同纠纷': '合同纠纷', 
    '网络购物合司纷': '合同纠纷', '电视购物合同纠纷': '合同纠纷', '招标投标买卖合同纠纷': '合同纠纷', 
    '拍卖合同纠纷': '合同纠纷', '建设用地使用权出让合同纠纷': '合同纠纷', '建设用地使用权转让合同纠纷': '合同纠纷',
    '临时用地合同纠纷': '合同纠纷', '探矿权转让合同纠纷': '合同纠纷', '采矿权转让合同纠纷': '合同纠纷', 
    '委托代建合同纠纷': '合同纠纷', '合资、合作开发房地产合同纠纷': '合同纠纷', '项目转让合同纠纷': '合同纠纷', 
    '商品房预约合同纠纷': '合同纠纷', '商品房预售合同纠纷': '合同纠纷', '商品房销售合同纠纷': '合同纠纷', 
    '商品房委托代理销售合同纠纷': '合同纠纷', '经济适用房转让合同纠纷': '合同纠纷', '农村房屋买卖合同纠纷': '合同纠纷', 
    '农村房屋买卖合同纠纷': '合同纠纷', '房屋拆迁安置补偿合纠纷': '合同纠纷', '供用电合同纠纷': '合同纠纷', 
    '供用水合同纠纷': '合同纠纷', '供用气合同纠纷': '合同纠纷', '供用热力合同纠纷': '合同纠纷', '公益事业捐赠合同纠纷': '合同纠纷', 
    '附义务赠与合同纠纷': '合同纠纷', '金融借款合同纠纷': '合同纠纷', '同业拆借纠纷': '合同纠纷', '企业借贷纠纷': '合同纠纷', 
    '民间借贷纠纷': '合同纠纷', '小额借款合同纠纷': '合同纠纷', '金融不良债权转让合同纠纷': '合同纠纷', 
    '保证合同纠纷': '合同纠纷', '抵押合同纠纷': '合同纠纷', '质押合同纠纷': '合同纠纷', '定金合同纠纷': '合同纠纷', 
    '进出口押汇纠纷': '合同纠纷', '储蓄存款合同纠纷': '合同纠纷', '借记卡纠纷': '合同纠纷', '信用卡纠纷': '合同纠纷', 
    '土地租赁合同纠纷': '合同纠纷', '房屋租赁合同纠纷': '合同纠纷', '车辆租赁合同纠纷': '合同纠纷', 
    '建筑设备租赁合同纠纷': '合同纠纷', '融资租赁合同纠纷': '合同纠纷', '加工合同纠纷': '合同纠纷', 
    '定作合同纠纷': '合同纠纷', '修理合同纠纷': '合同纠纷', '复制合同纠纷': '合同纠纷', '测试合同纠纷': '合同纠纷', 
    '检验合同纠纷': '合同纠纷', '铁路机车、车辆建造合同纠纷': '合同纠纷', '建设工程勘察合同纠纷': '合同纠纷', 
    '建设工程设计合同纠纷': '合同纠纷', '建设工程施工合同纠纷': '合同纠纷', '建设工程价款优先受偿权纠纷': '合同纠纷', 
    '建设工程分包合同纠纷': '合同纠纷', '建设工程监理合同纠纷': '合同纠纷', '装饰装修合同纠纷': '合同纠纷', 
    '铁路修建合同纠纷': '合同纠纷', '农村建房施工合同纠纷': '合同纠纷', '': '合同纠纷', '公路旅客运输合同纠纷': '合同纠纷', 
    '公路货物运输合同纠纷': '合同纠纷', '水路旅客运输合同纠纷': '合同纠纷', '水路货物运输合同纠纷': '合同纠纷', 
    '航空旅客运输合同纠纷': '合同纠纷', '航空货物运输合同纠纷': '合同纠纷', '': '合同纠纷', '出租汽车运输合同纠纷': '合同纠纷', 
    '管道运输合同纠纷': '合同纠纷', '城市公交运输合同纠纷': '合同纠纷', '联合运输合同纠纷': '合同纠纷', 
    '多式联运合同纠纷': '合同纠纷', '铁路旅客运输合同纠纷': '合同纠纷', '铁路行李运输合同纠纷': '合同纠纷', 
    '铁路包裹运输合同纠纷': '合同纠纷', '国际铁路联运合同纠纷': '合同纠纷', '保管合同纠纷': '合同纠纷', '仓储合同纠纷': '合同纠纷', 
    '进出口代理合同纠纷': '合同纠纷', '货运代理合同纠纷': '合同纠纷', '民用航空运输销售代理合同纠纷': '合同纠纷', 
    '诉讼、仲裁、人民调解代理合同纠纷': '合同纠纷', '金融委托理财合同纠纷': '合同纠纷', '民间委托理财合同纠纷': '合同纠纷', 
    '行纪合同纠纷': '合同纠纷', '居间合同纠纷': '合同纠纷', '补偿贸易纠纷': '合同纠纷', '借用合同纠纷': '合同纠纷', '典当纠纷': '合同纠纷', 
    '合伙协议纠纷': '合同纠纷', '种植、养殖回收合同纠纷': '合同纠纷', '彩票、奖券纠纷': '合同纠纷', '中外合作勘探开发自然资源合同纠纷': '合同纠纷', 
    '农业承包合同纠纷': '合同纠纷', '林业承包合同纠纷': '合同纠纷', '渔业承包合同纠纷': '合同纠纷', '牧业承包合同纠纷': '合同纠纷', 
    '土地承包经营权转包合同纠纷': '合同纠纷', '土地承包经营权转让合同纠纷': '合同纠纷', '土地承包经营权互换合同纠纷': '合同纠纷', 
    '土地承包经营权入股合同纠纷': '合同纠纷', '土地承包经营权抵押合同纠纷': '合同纠纷', '土地承包经营权出租合同纠纷': '合同纠纷', 
    '电信服务合同纠纷': '合同纠纷', '邮寄服务合同纠纷': '合同纠纷', '医疗服务合同纠纷': '合同纠纷', '法律服务合同纠纷': '合同纠纷', 
    '旅游合同纠纷': '合同纠纷', '房地产咨询合同纠纷': '合同纠纷', '房地产价格评估合同纠纷': '合同纠纷', '酒店服务合同纠纷': '合同纠纷', 
    '财会服务合同纠纷': '合同纠纷', '餐饮服务合同纠纷': '合同纠纷', '娱乐服务合同纠纷': '合同纠纷', '有线电视服务合同纠纷': '合同纠纷', 
    '网络服务合同纠纷': '合同纠纷', '教育培训合同纠纷': '合同纠纷', '物业服务合同纠纷': '合同纠纷', '家政服务合同纠纷': '合同纠纷', 
    '庆典服务合同纠纷': '合同纠纷', '殡葬服务合同纠纷': '合同纠纷', '农业技术服务合同纠纷': '合同纠纷', '农机作业服务合同纠纷': '合同纠纷', 
    '保安服务合同纠纷': '合同纠纷', '银行结算合同纠纷': '合同纠纷', '演出合同纠纷': '合同纠纷', '劳务合同纠纷': '合同纠纷', 
    '离退休人员返聘合同纠纷': '合同纠纷', '广告合同纠纷': '合同纠纷', '展览合同纠纷': '合同纠纷', '追偿权纠纷': '合同纠纷', '金融不良债权追偿纠纷': '合同纠纷',
    '铁路货物运输合同纠纷': '合同纠纷',

    '企业出资人权益确认纠纷': '与公司、证券、保险、票据等有关的民事纠纷','侵害企业出资人权益纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '企业公司制改造合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','企业股份合作制改造合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '企业债权转股权合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','企业分立合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '企业租赁经营合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','企业出售合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '挂靠经营合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','企业兼并合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '联营合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','中外合资经营企业承包经营合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '中外合作经营企业承包经营合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','外商独资企业承包经营合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '乡镇企业承包经营合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','中外合资经营企业合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '中外合作经营企业合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','股东资格确认纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '股东名册记载纠纷': '与公司、证券、保险、票据等有关的民事纠纷','请求变更公司登记纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '股东出资纠纷': '与公司、证券、保险、票据等有关的民事纠纷','新增资本认购纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '股东知情权纠纷': '与公司、证券、保险、票据等有关的民事纠纷','请求公司收购股份纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '股权转让纠纷': '与公司、证券、保险、票据等有关的民事纠纷','公司决议效力确认纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '公司决议撤销纠纷': '与公司、证券、保险、票据等有关的民事纠纷','公司设立纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '公司证照返还纠纷': '与公司、证券、保险、票据等有关的民事纠纷','发起人责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '公司盈余分配纠纷': '与公司、证券、保险、票据等有关的民事纠纷','损害股东利益责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '损害公司利益责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷','股东损害公司债权人利益责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '公司关联交易损害责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷','公司合并纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '公司分立纠纷': '与公司、证券、保险、票据等有关的民事纠纷','公司减资纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '公司增资纠纷': '与公司、证券、保险、票据等有关的民事纠纷','公司解散纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '申请公司清算': '与公司、证券、保险、票据等有关的民事纠纷','清算责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '上市公司收购纠纷': '与公司、证券、保险、票据等有关的民事纠纷','入伙纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '退伙纠纷': '与公司、证券、保险、票据等有关的民事纠纷','合伙企业财产份额转让纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '申请破产清算': '与公司、证券、保险、票据等有关的民事纠纷','申请破产重整': '与公司、证券、保险、票据等有关的民事纠纷',
    '申请破产和解': '与公司、证券、保险、票据等有关的民事纠纷','请求撤销个别清偿行为纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '请求确认债务人行为无效纠纷': '与公司、证券、保险、票据等有关的民事纠纷','对外追收债权纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '追收未缴出资纠纷': '与公司、证券、保险、票据等有关的民事纠纷','追收抽逃出资纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '追收非正常收入纠纷': '与公司、证券、保险、票据等有关的民事纠纷','职工破产债权确认纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '普通破产债权确认纠纷': '与公司、证券、保险、票据等有关的民事纠纷','一般取回权纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '出卖人取回权纠纷': '与公司、证券、保险、票据等有关的民事纠纷','破产抵销权纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '别除权纠纷': '与公司、证券、保险、票据等有关的民事纠纷','破产撤销权纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '损害债务人利益赔偿纠纷': '与公司、证券、保险、票据等有关的民事纠纷','管理人责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '股票权利确认纠纷': '与公司、证券、保险、票据等有关的民事纠纷','公司债券权利确认纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '国债权利确认纠纷': '与公司、证券、保险、票据等有关的民事纠纷','证券投资基金权利确认纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '股票交易纠纷': '与公司、证券、保险、票据等有关的民事纠纷','公司债券交易纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '国债交易纠纷': '与公司、证券、保险、票据等有关的民事纠纷','证券投资基金交易纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '金融衍生品种交易纠纷': '与公司、证券、保险、票据等有关的民事纠纷','证券代销合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '证券包销合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','证券投资咨询纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '证券资信评级服务合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','股票回购合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '国债回购合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','公司债券回购合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '证券投资基金回购合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','质押式证券回购纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '证券上市合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','证券交易代理合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '证券上市保荐合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','证券认购纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '证券发行失败纠纷': '与公司、证券、保险、票据等有关的民事纠纷','证券返还纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '证券内幕交易责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷','操纵证券交易市场责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '证券虚假陈述责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷','欺诈客户责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '证券托管纠纷': '与公司、证券、保险、票据等有关的民事纠纷','证券登记、存管、结算纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '融资融券交易纠纷': '与公司、证券、保险、票据等有关的民事纠纷','客户交易结算资金纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '期货经纪合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','期货透支交易纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '期货强行平仓纠纷': '与公司、证券、保险、票据等有关的民事纠纷','期货实物交割纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '期货保证合约纠纷': '与公司、证券、保险、票据等有关的民事纠纷','期货交易代理合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '侵占期货交易保证金纠纷': '与公司、证券、保险、票据等有关的民事纠纷','期货欺诈责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '操纵期货交易市场责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷','期货内幕交易责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '期货虚假信息责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷','民事信托纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '营业信托纠纷': '与公司、证券、保险、票据等有关的民事纠纷','公益信托纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '财产损失保险合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','责任保险合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '信用保险合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','保证保险合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '保险人代位求偿权纠纷': '与公司、证券、保险、票据等有关的民事纠纷','人寿保险合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '意外伤害保险合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','健康保险合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '再保险合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','保险经纪合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '保险代理合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷','进出口信用保险合同纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '保险费纠纷': '与公司、证券、保险、票据等有关的民事纠纷','票据付款请求权纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '票据追索权纠纷': '与公司、证券、保险、票据等有关的民事纠纷','票据交付请求权纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '票据返还请求权纠纷': '与公司、证券、保险、票据等有关的民事纠纷','票据损害责任纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '票据利益返还请求权纠纷': '与公司、证券、保险、票据等有关的民事纠纷','汇票回单签发请求权纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '票据保证纠纷': '与公司、证券、保险、票据等有关的民事纠纷','确认票据无效纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '票据代理纠纷': '与公司、证券、保险、票据等有关的民事纠纷','票据回购纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '委托开立信用证纠纷': '与公司、证券、保险、票据等有关的民事纠纷','信用证开证纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '信用证议付纠纷': '与公司、证券、保险、票据等有关的民事纠纷','信用证欺诈纠纷': '与公司、证券、保险、票据等有关的民事纠纷',
    '信用证融资纠纷': '与公司、证券、保险、票据等有关的民事纠纷','信用证转让纠纷': '与公司、证券、保险、票据等有关的民事纠纷'

}